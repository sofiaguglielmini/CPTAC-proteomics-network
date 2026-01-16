import pandas as pd
import numpy as np
import pickle
import requests
import mygene
from pathlib import Path
from sklearn.impute import KNNImputer
from scipy.stats import kurtosis


def load_cptac(path):
    df = pd.read_csv(path, sep='\t')
    genes = df.iloc[:, 0].str.replace(r'\..*$', '', regex=True)
    X = df.iloc[:, 1:].values.T
    sample_names = df.columns[1:].tolist()
    return pd.DataFrame(X, index=sample_names, columns=genes)

def preprocess(X, missing_threshold=0.2, kurtosis_threshold=10):
    X = X.loc[:, X.isna().mean() <= missing_threshold]
    
    imputer = KNNImputer(n_neighbors=10)
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)
    
    X = X.loc[:, X.var(axis=0) > 1e-10]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    gene_kurt = X.apply(kurtosis, axis=0)
    X = X.loc[:, gene_kurt <= kurtosis_threshold]
    
    return X


def load_and_process(): 
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'Proteome_BCM_GENCODE_v34_harmonized_v1'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    X_norm = load_cptac(data_dir / 'LSCC_proteomics_gene_abundance_log2_reference_intensity_normalized_Normal.txt')
    X_tumor = load_cptac(data_dir / 'LSCC_proteomics_gene_abundance_log2_reference_intensity_normalized_Tumor.txt')
    
    n_norm = len(X_norm)
    X_all = pd.concat([X_norm, X_tumor], axis=0)
    # Convert Ensembl IDs to gene symbols 
    ensembl_ids = X_all.columns.tolist()
    mg = mygene.MyGeneInfo()
    try:
        results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')
        symbol_mapping = {r['query']: r['symbol'] for r in results if 'symbol' in r and 'query' in r}
    except Exception:
        symbol_mapping = {}

    X_all.columns = [symbol_mapping.get(eid, eid) for eid in ensembl_ids]

    # Fetch KEGG cancer pathway genes
    import re
    kegg_genes = []
    try:
        text = requests.get("http://rest.kegg.jp/get/hsa05200", timeout=10).text
        entrez_ids = list(dict.fromkeys(re.findall(r'(?m)^\s*(\d+)\s', text)))
        if entrez_ids:
            kegg_results = mg.querymany(entrez_ids, scopes='entrezgene', fields='symbol', species='human')
            kegg_genes = [r['symbol'] for r in kegg_results if r.get('symbol')]
    except Exception:
        kegg_genes = []

    X_all = X_all[[col for col in X_all.columns if col in kegg_genes]]
    
    X_all = preprocess(X_all)
    
    X_norm_processed = X_all.iloc[:n_norm]
    X_tumor_processed = X_all.iloc[n_norm:]
    
    data = {
        'X_all': X_all,
        'X_norm': X_norm_processed,
        'X_tumor': X_tumor_processed
    }
    
    with open(results_dir / 'data_processed.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Processed: {X_all.shape[0]} samples, {X_all.shape[1]} genes")
    print(f"  Normal: {len(X_norm_processed)}, Tumor: {len(X_tumor_processed)}")


if __name__ == '__main__':
    load_and_process()
