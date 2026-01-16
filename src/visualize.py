import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import community as community_louvain


sns.set_style("white")
np.random.seed(42)

def detect_clusters(G):
    if len(G) == 0:
        return {}

    node_to_cluster = {}
    global_cluster_id = 0

    # Assign -1 to isolated nodes and run Louvain on components with >1 node. 
    # Change global_cluster_id to ensure unique cluster ids
    for component in nx.connected_components(G):
        if len(component) == 1:
            node = next(iter(component))
            node_to_cluster[node] = -1
            continue

        partition = community_louvain.best_partition(G.subgraph(component), randomize=False)
        local_to_global = {}
        for node, local_label in partition.items():
            if local_label not in local_to_global:
                local_to_global[local_label] = global_cluster_id
                global_cluster_id += 1
            node_to_cluster[node] = local_to_global[local_label]

    return node_to_cluster


def plot_network(pc, title, output_path, pos=None, palette='husl'):
    genes = list(pc.columns)
    pc_vals = pc.values
    
    G = nx.Graph()
    G.add_nodes_from(genes)
    
    for i in range(len(pc_vals)):
        for j in range(i + 1, len(pc_vals)):
            w = abs(pc_vals[i, j])
            if w > 0:
                G.add_edge(genes[i], genes[j], weight=w)
    
    clusters = detect_clusters(G)
    
    if pos is None:
        pos = nx.spring_layout(G, k=0.8 / np.sqrt(len(G.nodes()) + 1), iterations=100, seed=42)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()] if G.number_of_edges() > 0 else []
    edge_widths = [1 + 3 * (w / (max(edge_weights) + 1e-6)) for w in edge_weights] if edge_weights else []
    nx.draw_networkx_edges(G, pos, width=edge_widths if edge_widths else 1, alpha=0.25, ax=ax)
    
    degrees = dict(G.degree())
    node_sizes = [100 + degrees[n] * 50 for n in G.nodes()]
    
    connected_clusters = sorted(set(c for c in clusters.values() if c >= 0))
    cluster_sizes = {c: sum(1 for v in clusters.values() if v == c) for c in connected_clusters}
    sorted_clusters = sorted(connected_clusters, key=lambda c: cluster_sizes[c], reverse=True)
    
    n_clusters = len(connected_clusters) if connected_clusters else 1
    colors_palette = sns.color_palette(palette, n_clusters)
    cluster_to_color_idx = {cluster_id: idx for idx, cluster_id in enumerate(sorted_clusters)}
    
    node_colors = []
    for n in G.nodes():
        cluster_id = clusters[n]
        if cluster_id == -1:
            node_colors.append((0.7, 0.7, 0.7))
        else:
            color_idx = cluster_to_color_idx.get(cluster_id, 0)
            node_colors.append(colors_palette[color_idx])
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85, ax=ax)
    
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax,
                           bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
    
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    return pos, clusters


def save_clusters(pc, clusters, output_path):
    genes = list(pc.columns)
    
    cluster_data = []
    for gene in genes:
        cluster_id = clusters[gene]
        connectivity = abs(pc[gene]).sum()
        cluster_data.append({
            'Gene': gene,
            'Cluster': cluster_id,
            'Connectivity': connectivity
        })
    
    df = pd.DataFrame(cluster_data)
    df = df.sort_values(['Cluster', 'Connectivity'], ascending=[True, False])
    df_output = df[['Gene', 'Cluster']].copy()

    internal_path = output_path.parent / f"{output_path.stem}_internal.csv"
    df_output.to_csv(internal_path, index=False)
    print(f"  Saved (internal): {internal_path}")

def visualize():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    figures_dir = results_dir / 'figures'
    tables_dir = results_dir / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'network_all.pkl', 'rb') as f:
        net_all = pickle.load(f)
    with open(results_dir / 'network_case_control.pkl', 'rb') as f:
        net_cc = pickle.load(f)
    with open(results_dir / 'data_processed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    pc_all = pd.DataFrame(net_all['partialcorr'], columns=data['X_all'].columns, index=data['X_all'].columns)
    pc_norm = pd.DataFrame(net_cc['normal']['partialcorr'], columns=data['X_norm'].columns, index=data['X_norm'].columns)
    pc_tumor = pd.DataFrame(net_cc['tumor']['partialcorr'], columns=data['X_tumor'].columns, index=data['X_tumor'].columns)
        
    pos_all, clusters_all = plot_network(pc_all, 'Gene Network (All Samples)', figures_dir / 'network_all.png')
    save_clusters(pc_all, clusters_all, tables_dir / 'clusters_all.csv')
    
    genes = list(pc_norm.columns)
    G_combined = nx.Graph()
    G_combined.add_nodes_from(genes)
    for i in range(len(pc_norm)):
        for j in range(i + 1, len(pc_norm)):
            if np.abs(pc_norm.iloc[i, j]) > 0 or np.abs(pc_tumor.iloc[i, j]) > 0:
                G_combined.add_edge(genes[i], genes[j])
    
    pos_shared = nx.spring_layout(G_combined, k=0.5, iterations=50, seed=42)
    
    pos_norm, clusters_norm = plot_network(pc_norm, 'Protein Network (Normal)', 
                                           figures_dir / 'network_normal.png', pos=pos_shared, palette='cool')
    save_clusters(pc_norm, clusters_norm, tables_dir / 'clusters_normal.csv')
    
    pos_tumor, clusters_tumor = plot_network(pc_tumor, 'Protein Network (Tumor)', 
                                             figures_dir / 'network_tumor.png', pos=pos_shared, palette='YlOrRd')
    save_clusters(pc_tumor, clusters_tumor, tables_dir / 'clusters_tumor.csv')


if __name__ == '__main__':
    visualize()
