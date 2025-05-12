import json
import networkx as nx
from pathlib import Path
import time
from collections import defaultdict
import heapq
from dataclasses import dataclass
from typing import Dict, Set, Any, Optional, Tuple, List
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

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
    return n1.get('purl') == n2.get('purl') and n1.get('type') == n2.get('type')

def edge_match(e1, e2):
    return e1.get('type') == e2.get('type')

def compute_parallel_ged(G1, G2, beam_width=10000, max_iterations=100000, timeout=600, num_processes=None) -> Tuple[Optional[int], float]:
    """
    Compute GED using parallel beam search across multiple CPU cores.
    
    Parameters:
    - beam_width: Number of states to keep in beam search per process
    - max_iterations: Maximum number of iterations per process
    - timeout: Maximum time in seconds to run (10 minutes)
    - num_processes: Number of processes to use (defaults to CPU count - 1)
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)  # Leave one core free for system

    def get_node_cost(n1, n2):
        return 0 if node_match(n1, n2) else 1

    def get_edge_cost(e1, e2):
        return 0 if edge_match(e1, e2) else 1

    def compute_lower_bound(state: State) -> int:
        min_node_cost = min(len(state.unmapped_G1), len(state.unmapped_G2))
        mapped_edges_G1 = sum(1 for u, v in G1.edges() 
                            if u in state.mapping and v in state.mapping)
        mapped_edges_G2 = sum(1 for u, v in G2.edges() 
                            if u in state.mapping.values() and v in state.mapping.values())
        min_edge_cost = abs(mapped_edges_G1 - mapped_edges_G2)
        return state.cost + min_node_cost + min_edge_cost

    def process_beam_search(process_id: int) -> Tuple[Optional[int], float]:
        # Initialize with different starting states for each process
        initial_state = State(
            mapping={},
            cost=0,
            unmapped_G1=set(G1.nodes()),
            unmapped_G2=set(G2.nodes())
        )
        
        beam = [(0, process_id, initial_state)]  # (cost, process_id, state)
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
            
            state_key = (frozenset(state.mapping.items()), 
                        frozenset(state.unmapped_G1), 
                        frozenset(state.unmapped_G2))
            if state_key in seen_states:
                continue
            seen_states.add(state_key)
            
            if not state.unmapped_G1 and not state.unmapped_G2:
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
                
                for u2, v2 in G2.edges():
                    u1 = next((n1 for n1, n2 in state.mapping.items() if n2 == u2), None)
                    v1 = next((n1 for n1, n2 in state.mapping.items() if n2 == v2), None)
                    if u1 is None or v1 is None or not G1.has_edge(u1, v1):
                        edge_cost += 1
                
                total_cost = current_cost + edge_cost
                best_cost = min(best_cost, total_cost)
                continue
            
            next_states = []
            
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
                    next_states.append((new_cost, process_id, new_state))
            
            if state.unmapped_G1:
                n1 = next(iter(state.unmapped_G1))
                new_state = State(
                    mapping=state.mapping.copy(),
                    cost=current_cost + 1,
                    unmapped_G1=state.unmapped_G1 - {n1},
                    unmapped_G2=state.unmapped_G2
                )
                next_states.append((current_cost + 1, process_id, new_state))
            
            if state.unmapped_G2:
                n2 = next(iter(state.unmapped_G2))
                new_state = State(
                    mapping=state.mapping.copy(),
                    cost=current_cost + 1,
                    unmapped_G1=state.unmapped_G1,
                    unmapped_G2=state.unmapped_G2 - {n2}
                )
                next_states.append((current_cost + 1, process_id, new_state))
            
            for state_tuple in next_states:
                cost, _, new_state = state_tuple
                lower_bound = compute_lower_bound(new_state)
                if lower_bound < best_cost:
                    heapq.heappush(beam, state_tuple)
            
            if len(beam) > beam_width:
                beam = beam[:beam_width]
                
            if current_time - last_progress_time >= 2:
                completion = min(100, (iteration / max_iterations) * 100)
                elapsed = current_time - start_time
                states_per_second = iteration / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\rProcess {process_id} - Progress: {completion:.1f}% | Best cost: {best_cost if best_cost != float('inf') else 'None'} | States/sec: {states_per_second:.1f} | Time: {elapsed:.1f}s")
                sys.stdout.flush()
                last_progress_time = current_time
        
        completion = min(100, (iteration / max_iterations) * 100)
        return (best_cost if best_cost != float('inf') else None, completion)

    print(f"\nStarting parallel GED computation with {num_processes} processes...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_beam_search, i) for i in range(num_processes)]
        results = []
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result[0] is not None:  # If we found a solution
                print(f"\nFound solution with cost: {result[0]}")
    
    # Get the best result from all processes
    best_result = min((r for r in results if r[0] is not None), key=lambda x: x[0], default=(None, 0))
    best_cost, completion = best_result
    
    total_time = time.time() - start_time
    print(f"\nParallel computation completed in {total_time:.2f} seconds")
    print(f"Best GED found: {best_cost}")
    print(f"Average completion: {sum(r[1] for r in results) / len(results):.1f}%")
    
    return best_result

def process_repo_comparison(repo_dir: Path, num_processes: int = None):
    """Process all graph comparisons in a repository using parallel computation."""
    json_dir = repo_dir / 'json'
    if not json_dir.exists():
        return None
        
    files = list(json_dir.glob('*.json'))
    if len(files) < 2:
        print(f"Not enough JSON files found in {json_dir} for comparison.")
        return None
        
    print(f"\nProcessing repo: {repo_dir.name}")
    graphs = [load_graph_from_json(str(f)) for f in files]
    print(f"Loaded {len(graphs)} graphs")
    
    results = []
    for i in range(len(graphs)):
        for j in range(i+1, len(graphs)):
            print(f"\nComparing graph {i+1} ({files[i].name}) with graph {j+1} ({files[j].name})...")
            print(f"Graph 1 stats: {len(graphs[i].nodes)} nodes, {len(graphs[i].edges)} edges")
            print(f"Graph 2 stats: {len(graphs[j].nodes)} nodes, {len(graphs[j].edges)} edges")
            
            start_time = time.time()
            ged, completion = compute_parallel_ged(graphs[i], graphs[j], num_processes=num_processes)
            elapsed = time.time() - start_time
            
            results.append({
                'file1': files[i].name,
                'file2': files[j].name,
                'ged': ged,
                'completion': completion,
                'computation_time': elapsed
            })
            
    return results

def main():
    base_dir = Path('graphoutput')
    all_results = {}
    
    # Get number of CPU cores to use
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} CPU cores for parallel computation")
    
    for repo_dir in base_dir.iterdir():
        if not repo_dir.is_dir():
            continue
            
        results = process_repo_comparison(repo_dir, num_processes)
        if results:
            all_results[repo_dir.name] = results
    
    # Save results to a JSON file
    output_file = base_dir / 'ged_parallel_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main() 