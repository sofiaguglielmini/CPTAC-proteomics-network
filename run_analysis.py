#!/usr/bin/env python3

import sys
from pathlib import Path
import pickle
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import load_and_process
from network import analyze_case_control, compute_pathway_metrics
from visualize import visualize


print("CPTAC Proteomics Network Analysis\n")

print("Step 1: Loading and preprocessing data...")
load_and_process()

print("\nStep 2: Analyzing networks...")
analyze_case_control()

print("\nStep 3: Creating visualizations...")
visualize()

print("\nStep 4: Computing pathway metrics...")
results_dir = Path(__file__).parent / 'results'
tables_dir = results_dir / 'tables'

with open(results_dir / 'network_case_control.pkl', 'rb') as f:
	net_cc = pickle.load(f)
with open(results_dir / 'data_processed.pkl', 'rb') as f:
	data = pickle.load(f)

pc_norm = pd.DataFrame(net_cc['normal']['partialcorr'], columns=data['X_norm'].columns, index=data['X_norm'].columns)
pc_tumor = pd.DataFrame(net_cc['tumor']['partialcorr'], columns=data['X_tumor'].columns, index=data['X_tumor'].columns)

metrics_norm = compute_pathway_metrics(pc_norm, tables_dir / 'clusters_normal_internal.csv')
metrics_tumor = compute_pathway_metrics(pc_tumor, tables_dir / 'clusters_tumor_internal.csv')

if metrics_norm and metrics_tumor:
	lscc_comparison = pd.DataFrame({
		'Metric': [
			'Mean_Weighted_Degree',
			'Total_Connectivity',
			'Mean_Betweenness',
			'Concentration_Index'
		],
		'Normal': [
			metrics_norm['mean_weighted_degree'],
			metrics_norm['total_connectivity'],
			metrics_norm['mean_betweenness'],
			metrics_norm['concentration_index']
		],
		'Tumor': [
			metrics_tumor['mean_weighted_degree'],
			metrics_tumor['total_connectivity'],
			metrics_tumor['mean_betweenness'],
			metrics_tumor['concentration_index']
		]
	})
	lscc_comparison['Difference'] = lscc_comparison['Tumor'] - lscc_comparison['Normal']
	lscc_comparison['Percent_Change'] = (lscc_comparison['Difference'] / lscc_comparison['Normal'] * 100).round(2)
	lscc_comparison.to_csv(tables_dir / 'lscc_pathway_metrics.csv', index=False)

print("\nAnalysis complete")
