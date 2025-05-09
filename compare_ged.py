import json
import networkx as nx
from pathlib import Path
import time

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

def print_all_node_matches(G1, G2):
    print("Node match attempts (by purl):")
    for n1, d1 in G1.nodes(data=True):
        for n2, d2 in G2.nodes(data=True):
            match = node_match(d1, d2)
            print(f"  {n1} <-> {n2}: {'MATCH' if match else 'NO MATCH'}")
    print()

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
    print("All graphs loaded. Starting GED comparisons...\n")

    # Initialize results for this repo
    results[repo_dir.name] = []

    # Compute pairwise GED (using attribute-aware matching) for graphs in the same repo
    for i in range(len(graphs)):
        for j in range(i+1, len(graphs)):
            print(f"Comparing graph {i+1} ({files[i].name}) with graph {j+1} ({files[j].name})...")
            start_time = time.time()
            ged = nx.graph_edit_distance(
                graphs[i], graphs[j],
                node_match=node_match,
                edge_match=edge_match
            )
            elapsed = time.time() - start_time
            print(f"GED (purl-based) between graph {i+1} and graph {j+1}: {ged}")
            print(f"Time spent: {elapsed:.2f} seconds\n")
            
            # Store the result
            results[repo_dir.name].append({
                'file1': files[i].name,
                'file2': files[j].name,
                'ged': ged,
                'computation_time': elapsed
            })

# Save results to a JSON file
output_file = base_dir / 'ged_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_file}") 