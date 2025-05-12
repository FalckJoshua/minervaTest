import json
import networkx as nx
from pathlib import Path
import time
from collections import defaultdict
import heapq
from dataclasses import dataclass
from typing import Dict, Set, Any, Optional, Tuple
import sys

@dataclass
class State:
    mapping: Dict[str, str]
    cost: int
    unmapped_G1: Set[str]
    unmapped_G2: Set[str]
    
    def __lt__(self, other):
        return self.cost < other.cost

def load_graph_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    G = nx.DiGraph()
    # Build a mapping from id to purl (or id if purl not present)
    id_to_purl = {}
    for node in data['nodes']:
        node_id = node['id']
        node_purl = node.get('purl', node_id)
        id_to_purl[node_id] = node_purl
        attrs = {k: v for k, v in node.items() if k != 'id'}
        G.add_node(node_purl, **attrs)
    for link in data['links']:
        source = id_to_purl[link['source']]
        target = id_to_purl[link['target']]
        attrs = {k: v for k, v in link.items() if k not in ['source', 'target']}
        G.add_edge(source, target, **attrs)
    return G

def node_match(n1, n2):
    # Compare 'purl' and 'type' if present
    return n1.get('purl') == n2.get('purl') and n1.get('type') == n2.get('type')

def edge_match(e1, e2):
    return e1.get('type') == e2.get('type')

def compute_optimized_ged(G1, G2, beam_width=5000, max_iterations=50000, timeout=300) -> Tuple[Optional[int], float]:
    """
    Compute GED using beam search and early pruning for efficiency.
    Returns a tuple of (minimum edit distance found, completion percentage).
    
    Parameters:
    - beam_width: Number of states to keep in beam search (increased for supercomputer)
    - max_iterations: Maximum number of iterations to perform
    - timeout: Maximum time in seconds to run (5 minutes)
    """
    def get_node_cost(n1, n2):
        return 0 if node_match(n1, n2) else 1

    def get_edge_cost(e1, e2):
        return 0 if edge_match(e1, e2) else 1

    def compute_lower_bound(state: State) -> int:
        """Compute a lower bound on the remaining cost for a state."""
        # Minimum cost for remaining node mappings
        min_node_cost = min(len(state.unmapped_G1), len(state.unmapped_G2))
        
        # Minimum cost for remaining edges
        mapped_edges_G1 = sum(1 for u, v in G1.edges() 
                            if u in state.mapping and v in state.mapping)
        mapped_edges_G2 = sum(1 for u, v in G2.edges() 
                            if u in state.mapping.values() and v in state.mapping.values())
        min_edge_cost = abs(mapped_edges_G1 - mapped_edges_G2)
        
        return state.cost + min_node_cost + min_edge_cost

    # Initialize with empty mapping
    initial_state = State(
        mapping={},
        cost=0,
        unmapped_G1=set(G1.nodes()),
        unmapped_G2=set(G2.nodes())
    )
    
    # Priority queue for beam search
    beam = [(0, 0, initial_state)]  # (cost, tiebreaker, state)
    seen_states = set()
    iteration = 0
    best_cost = float('inf')
    start_time = time.time()
    last_progress_time = start_time
    
    while beam and iteration < max_iterations:
        current_time = time.time()
        if current_time - start_time > timeout:
            break
            
        iteration += 1
        current_cost, _, state = heapq.heappop(beam)
        
        # Skip if we've seen this state
        state_key = (frozenset(state.mapping.items()), 
                    frozenset(state.unmapped_G1), 
                    frozenset(state.unmapped_G2))
        if state_key in seen_states:
            continue
        seen_states.add(state_key)
        
        # If all nodes are mapped, compute final cost
        if not state.unmapped_G1 and not state.unmapped_G2:
            # Compute edge costs for the complete mapping
            edge_cost = 0
            for u1, v1 in G1.edges():
                u2 = state.mapping.get(u1)
                v2 = state.mapping.get(v1)
                if u2 is not None and v2 is not None:
                    e1 = G1[u1][v1]
                    e2 = G2[u2][v2] if G2.has_edge(u2, v2) else None
                    edge_cost += get_edge_cost(e1, e2) if e2 else 1
                else:
                    edge_cost += 1
            
            # Add costs for edges in G2 that aren't in the mapping
            for u2, v2 in G2.edges():
                u1 = next((n1 for n1, n2 in state.mapping.items() if n2 == u2), None)
                v1 = next((n1 for n1, n2 in state.mapping.items() if n2 == v2), None)
                if u1 is None or v1 is None or not G1.has_edge(u1, v1):
                    edge_cost += 1
            
            total_cost = current_cost + edge_cost
            best_cost = min(best_cost, total_cost)
            continue
        
        # Generate next states
        next_states = []
        
        # Try mapping an unmapped node from G1 to an unmapped node from G2
        if state.unmapped_G1 and state.unmapped_G2:
            n1 = next(iter(state.unmapped_G1))
            for n2 in state.unmapped_G2:
                new_mapping = state.mapping.copy()
                new_mapping[n1] = n2
                new_cost = current_cost + get_node_cost(G1.nodes[n1], G2.nodes[n2])
                
                new_state = State(
                    mapping=new_mapping,
                    cost=new_cost,
                    unmapped_G1=state.unmapped_G1 - {n1},
                    unmapped_G2=state.unmapped_G2 - {n2}
                )
                next_states.append((new_cost, iteration, new_state))
        
        # Try deleting a node from G1
        if state.unmapped_G1:
            n1 = next(iter(state.unmapped_G1))
            new_state = State(
                mapping=state.mapping.copy(),
                cost=current_cost + 1,  # Node deletion cost
                unmapped_G1=state.unmapped_G1 - {n1},
                unmapped_G2=state.unmapped_G2
            )
            next_states.append((current_cost + 1, iteration, new_state))
        
        # Try inserting a node from G2
        if state.unmapped_G2:
            n2 = next(iter(state.unmapped_G2))
            new_state = State(
                mapping=state.mapping.copy(),
                cost=current_cost + 1,  # Node insertion cost
                unmapped_G1=state.unmapped_G1,
                unmapped_G2=state.unmapped_G2 - {n2}
            )
            next_states.append((current_cost + 1, iteration, new_state))
        
        # Add next states to beam with pruning
        for state_tuple in next_states:
            cost, _, new_state = state_tuple
            lower_bound = compute_lower_bound(new_state)
            if lower_bound < best_cost:  # Early pruning
                heapq.heappush(beam, state_tuple)
        
        # Keep beam size limited
        if len(beam) > beam_width:
            beam = beam[:beam_width]
            
        # Print progress every 2 seconds
        if current_time - last_progress_time >= 2:
            completion = min(100, (iteration / max_iterations) * 100)
            elapsed = current_time - start_time
            states_per_second = iteration / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\rProgress: {completion:.1f}% | Best cost: {best_cost if best_cost != float('inf') else 'None'} | States/sec: {states_per_second:.1f} | Time: {elapsed:.1f}s")
            sys.stdout.flush()
            last_progress_time = current_time
    
    print()  # New line after progress
    completion = min(100, (iteration / max_iterations) * 100)
    return (best_cost if best_cost != float('inf') else None, completion)

# Paths to your JSON files for all repos
base_dir = Path('graphoutput')
results = {}  # Dictionary to store all results

for repo_dir in base_dir.iterdir():
    if not repo_dir.is_dir():
        continue
    json_dir = repo_dir / 'json'
    if not json_dir.exists():
        continue
    files = list(json_dir.glob('*.json'))
    if len(files) < 2:
        print(f"Not enough JSON files found in {json_dir} for comparison.")
        continue
    print(f"Processing repo: {repo_dir.name}")
    # Load graphs
    for idx, f in enumerate(files):
        print(f"Loading graph {idx+1} from: {f}")
    graphs = [load_graph_from_json(str(f)) for f in files]
    print("All graphs loaded. Starting optimized GED comparisons...\n")

    # Initialize results for this repo
    results[repo_dir.name] = []

    # Compute pairwise GED for graphs in the same repo
    for i in range(len(graphs)):
        for j in range(i+1, len(graphs)):
            print(f"Comparing graph {i+1} ({files[i].name}) with graph {j+1} ({files[j].name})...")
            print(f"Graph 1 stats: {len(graphs[i].nodes)} nodes, {len(graphs[i].edges)} edges")
            print(f"Graph 2 stats: {len(graphs[j].nodes)} nodes, {len(graphs[j].edges)} edges")
            print("\nStarting optimized GED computation...\n")
            start_time = time.time()
            
            ged, completion = compute_optimized_ged(graphs[i], graphs[j])
            
            elapsed = time.time() - start_time
            print(f"Optimized GED between graph {i+1} and graph {j+1}: {ged}")
            print(f"Completion: {completion:.1f}%")
            print(f"Time spent: {elapsed:.2f} seconds\n")
            
            # Store the result
            results[repo_dir.name].append({
                'file1': files[i].name,
                'file2': files[j].name,
                'ged': ged,
                'completion': completion,
                'computation_time': elapsed
            })

# Save results to a JSON file
output_file = base_dir / 'ged_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_file}") 