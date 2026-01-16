import numpy as np
import pandas as pd
import pickle
import mygene
import requests
import networkx as nx
from pathlib import Path
from sklearn.covariance import GraphicalLasso
from sklearn.utils import resample
import matplotlib.pyplot as plt
import warnings

def estimate_network(X, alpha=None):
    X_vals = X.values if hasattr(X, 'values') else X
    
    if alpha is None:
        n_samples, n_features = X_vals.shape
        alpha = np.sqrt(np.log(n_features) / n_samples) * 2.5
    gl = GraphicalLasso(alpha=alpha, max_iter=1000, tol=1e-3, assume_centered=True)
    with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.linalg")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.linalg")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
            gl.fit(X_vals)    
    prec = gl.precision_
    d = np.diag(prec)
    pc = -prec / np.sqrt(np.outer(d, d))
    np.fill_diagonal(pc, 1)
    
    return pd.DataFrame(pc, index=X.columns, columns=X.columns)


def compute_network_stats(pc):
    pc_vals = pc.values.copy()
    np.fill_diagonal(pc_vals, 0)
    
    edges = (np.abs(pc_vals) > 0).sum(axis=1)
    n_edges = edges.sum() // 2
    n_genes = len(pc)
    
    return {
        'prop_edges': n_edges / (n_genes * (n_genes - 1) / 2),
        'avg_degree': edges.mean() / 2,
        'avg_weighted_degree': np.abs(pc_vals).sum(axis=1).mean() / 2
    }

def bootstrap_hubs(X, n_bootstrap=100, frac=0.8, seed=1):
    np.random.seed(seed)
    n, p = X.shape
    wdeg = np.zeros((p, n_bootstrap))
    
    for b in range(n_bootstrap):
        X_boot = resample(X, n_samples=int(frac * n), random_state=seed + b)
        pc = estimate_network(X_boot)
        np.fill_diagonal(pc.values, 0)
        wdeg[:, b] = pc.abs().sum(axis=1).values
    
    return wdeg


def annotate_genes(gene_symbols):
    mg = mygene.MyGeneInfo()
    # Query MyGene for human gene names; return mapping symbol -> full name
    try:
        results = mg.querymany(list(gene_symbols), scopes='symbol', fields='name', species='human')
    except Exception:
        results = []

    annotations = {}
    for r in results:
        q = r.get('query')
        name = r.get('name')
        if q and name:
            annotations[q] = name

    return annotations

def compute_pathway_metrics(pc, clusters_csv_path):
    import re

    # Get KEGG genes in Non-small cell lung cancer pathway
    text = requests.get("http://rest.kegg.jp/get/hsa05223").text
    entrez_ids = re.findall(r'(?m)^\s*(\d+)\s', text)
    mg = mygene.MyGeneInfo()
    results = mg.querymany(entrez_ids, scopes='entrezgene', fields='symbol', species='human')
    lscc_genes = set(r['symbol'] for r in results if 'symbol' in r)

    clusters_df = pd.read_csv(clusters_csv_path)

    pathway_genes = [g for g in lscc_genes if g in pc.columns]
    weighted_degrees = pc[pathway_genes].abs().sum(axis=1)
    mean_weighted_degree = weighted_degrees.mean()
    total_connectivity = weighted_degrees.sum()

    pc_vals = pc.values.copy()
    np.fill_diagonal(pc_vals, 0)
    G = nx.Graph()
    genes = list(pc.columns)
    G.add_nodes_from(genes)
    for i in range(len(pc_vals)):
        for j in range(i + 1, len(pc_vals)):
            if abs(pc_vals[i, j]) > 0:
                G.add_edge(genes[i], genes[j], weight=abs(pc_vals[i, j]))
    betweenness = nx.betweenness_centrality(G, weight='weight')
    mean_betweenness = pd.Series(betweenness)[pathway_genes].mean()

    # Herfindahl index (sum of squares of share of pathway genes in each cluster)
    pathway_genes_clusters = clusters_df.loc[clusters_df['Gene'].isin(pathway_genes), 'Cluster']
    counts = pathway_genes_clusters[pathway_genes_clusters >= 0].value_counts() # how many pathway genes per cluster
    concentration = ((counts / len(pathway_genes)) ** 2).sum() 

    return {
        'mean_weighted_degree': mean_weighted_degree,
        'total_connectivity': total_connectivity,
        'mean_betweenness': mean_betweenness,
        'concentration_index': concentration,
        'n_genes': len(pathway_genes)
    }

def analyze_case_control():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    tables_dir = results_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'data_processed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_norm = data['X_norm']
    X_tumor = data['X_tumor']
    X_all = data.get('X_all')
    
    print("Estimating full, normal and tumor networks...")
    # Full network (all samples)
    if X_all is not None:
        pc_all = estimate_network(X_all)
        with open(results_dir / 'network_all.pkl', 'wb') as f:
            pickle.dump({'partialcorr': pc_all.values}, f)
        stats_all = compute_network_stats(pc_all)
        summary_all = pd.DataFrame({
            'Metric': ['Proportion_Nonzero_Edges', 'Average_Node_Degree', 'Mean_Weighted_Degree'],
            'All': [stats_all['prop_edges'], stats_all['avg_degree'], stats_all['avg_weighted_degree']]
        })
        summary_all.to_csv(tables_dir / 'network_summary_all.csv', index=False)

    pc_norm = estimate_network(X_norm)
    pc_tumor = estimate_network(X_tumor)
    
    stats_norm = compute_network_stats(pc_norm)
    stats_tumor = compute_network_stats(pc_tumor)
    
    summary = pd.DataFrame({
        'Metric': ['Proportion_Nonzero_Edges', 'Average_Node_Degree', 'Mean_Weighted_Degree'],
        'Normal': [stats_norm['prop_edges'], stats_norm['avg_degree'], stats_norm['avg_weighted_degree']],
        'Tumor': [stats_tumor['prop_edges'], stats_tumor['avg_degree'], stats_tumor['avg_weighted_degree']]
    })
    summary.to_csv(tables_dir / 'network_summary_case_control.csv', index=False)
    
    with open(results_dir / 'network_case_control.pkl', 'wb') as f:
        pickle.dump({
            'normal': {'partialcorr': pc_norm.values},
            'tumor': {'partialcorr': pc_tumor.values}
        }, f)
    
    print("Computing hub stability...")
    wdeg_norm = bootstrap_hubs(X_norm)
    wdeg_tumor = bootstrap_hubs(X_tumor)
    diff_wdeg = wdeg_tumor - wdeg_norm
    
    hub_summary = pd.DataFrame({
        'Gene': X_norm.columns,
        'MWD_Normal': wdeg_norm.mean(axis=1),
        'MWD_Tumor': wdeg_tumor.mean(axis=1),
        'Difference': wdeg_tumor.mean(axis=1) - wdeg_norm.mean(axis=1),
        'SD_Difference': diff_wdeg.std(axis=1)
    })

    sd_norm = wdeg_norm.std(axis=1)
    sd_tumor = wdeg_tumor.std(axis=1)
    mean_norm = wdeg_norm.mean(axis=1)
    mean_tumor = wdeg_tumor.mean(axis=1)
    cv_norm = sd_norm / np.where(mean_norm == 0, np.nan, mean_norm) 
    cv_tumor = sd_tumor / np.where(mean_tumor == 0, np.nan, mean_tumor)

    hub_summary['SD_Normal'] = sd_norm
    hub_summary['SD_Tumor'] = sd_tumor
    hub_summary['CV_Normal'] = cv_norm
    hub_summary['CV_Tumor'] = cv_tumor
    
    hub_summary = hub_summary.sort_values('Difference', key=abs, ascending=False)
    
    gene_names = annotate_genes(hub_summary['Gene'].head(10).tolist())
    hub_summary.loc[hub_summary.index[:10], 'Gene_Name'] = (hub_summary['Gene'].head(10).map(gene_names).fillna('N/A'))
    
    hub_summary.to_csv(tables_dir / 'hub_comparison_case_control.csv', index=False)
    top_n = 20 
    
    # Annotate all top genes for normal and tumor
    top_norm = hub_summary[['Gene', 'MWD_Normal', 'CV_Normal']].copy()
    top_norm = top_norm.rename(columns={'MWD_Normal': 'Mean_Weighted_Degree', 'CV_Normal': 'CV'})
    top_norm = top_norm.sort_values('Mean_Weighted_Degree', ascending=False).head(top_n)
    norm_gene_names = annotate_genes(top_norm['Gene'].tolist())
    top_norm['Gene_Name'] = top_norm['Gene'].map(norm_gene_names).fillna('N/A')
    top_norm.to_csv(tables_dir / 'top_hubs_normal.csv', index=False)

    top_tumor = hub_summary[['Gene', 'MWD_Tumor', 'CV_Tumor']].copy()
    top_tumor = top_tumor.rename(columns={'MWD_Tumor': 'Mean_Weighted_Degree', 'CV_Tumor': 'CV'})
    top_tumor = top_tumor.sort_values('Mean_Weighted_Degree', ascending=False).head(top_n)
    tumor_gene_names = annotate_genes(top_tumor['Gene'].tolist())
    top_tumor['Gene_Name'] = top_tumor['Gene'].map(tumor_gene_names).fillna('N/A')
    top_tumor.to_csv(tables_dir / 'top_hubs_tumor.csv', index=False)
    
    # edge differences for all gene pairs
    edges = []
    genes = X_norm.columns
    pc_norm_vals = pc_norm.values
    pc_tumor_vals = pc_tumor.values
    
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            edges.append({
                'Gene1': genes[i],
                'Gene2': genes[j],
                'PartialCorr_Normal': pc_norm_vals[i, j],
                'PartialCorr_Tumor': pc_tumor_vals[i, j],
                'Difference': pc_tumor_vals[i, j] - pc_norm_vals[i, j]
            })
    
    edge_tbl = pd.DataFrame(edges)
    edge_tbl.to_csv(tables_dir / 'edge_differences.csv', index=False)
    
    # Plot top edge differences, largest first
    edge_tbl['Abs_Difference'] = edge_tbl['Difference'].abs()
    top_edges = edge_tbl.nlargest(20, 'Abs_Difference')
    top_edges = top_edges.sort_values('Abs_Difference')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    edge_labels = top_edges['Gene1'] + ' - ' + top_edges['Gene2']
    colors = ['red' if d > 0 else 'blue' for d in top_edges['Difference']]
    ax.barh(range(len(top_edges)), top_edges['Difference'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_edges)))
    ax.set_yticklabels(edge_labels, fontsize=9)
    ax.set_xlabel('Partial Correlation Difference (Tumor - Normal)', fontsize=11)
    ax.set_title('Top 20 Edge Changes (by absolute value)', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'top_edge_differences.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figures_dir / 'top_edge_differences.png'}")

if __name__ == '__main__':
    analyze_case_control()
