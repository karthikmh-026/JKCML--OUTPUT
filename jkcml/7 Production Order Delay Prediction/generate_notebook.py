"""
Generates a 100% CODE-ONLY Jupyter Notebook for Use Case 7: Production Order Delay Prediction.
Every single cell is an executable Python code cell across the 21 steps.
Operates on ALL combined data across HANA_P2 and HANA_P3 schemas without row limits.
"""
import os
import nbformat as nbf

def create_code_only_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 01. Business Problem
    cells.append(nbf.v4.new_code_cell(r"""# 01. Business Problem & Project Overview
USE_CASE = "07 - Production Order Delay Prediction (Predict Delayed Production Orders)"
ML_PARADIGM = "Unsupervised Machine Learning (5 Models Evaluated & Ranked -> #1 Champion Model Selected)"
SCHEMAS = "HANA_P2 and HANA_P3 (All Combined Production Order Data - NO LIMIT)"
N_CLUSTERS = 4

print(f"Business Problem: {USE_CASE}")
print(f"ML Approach:      {ML_PARADIGM}")
print(f"Database Schemas: {SCHEMAS}")
print(f"Delay Risk Tiers: {N_CLUSTERS} Tiers (On-Time, Low Risk, Moderate Risk, Critical Delay Exception)")
print("Objective:        Predict delayed production orders and estimate calibrated Probability of Delay P(Delay)")
"""))

    # 02. Import Libraries
    cells.append(nbf.v4.new_code_cell(r"""# 02. Import Libraries & Set Global Environments
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve,
    silhouette_score
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)
print("All core ML, clustering, and data processing packages successfully imported.")
"""))

    # 03. Connect to Server
    cells.append(nbf.v4.new_code_cell(r"""# 03. Database Server Connection & Configuration
from src.db_connector import load_config, get_db_connection

config = load_config()
db = config["database"]
print(f"Database Server: {db['host']}:{db['port']}")
print(f"Database Name:   {db['dbname']}")
print(f"Schemas:         {db.get('schemas', ['HANA_P2', 'HANA_P3'])}")

conn = get_db_connection(config)
print("Live PostgreSQL connection established successfully.")
conn.close()
"""))

    # 04. Extract Data
    cells.append(nbf.v4.new_code_cell(r"""# 04. Extract All Combined Production Order Data (NO LIMIT)
from src.db_connector import extract_production_order_data

raw_df = extract_production_order_data(limit=None, config=config, use_cache=True)
print(f"Extracted Production Order DataFrame Shape: {raw_df.shape}")
print(f"Total Columns: {len(raw_df.columns)}")
"""))

    # 05. Schema Inspection
    cells.append(nbf.v4.new_code_cell(r"""# 05. Exploratory Data Analysis & Schema Inspection
print("Dataset Summary:")
print(raw_df[['order_number', 'order_type', 'plant', 'planned_start_date', 'planned_finish_date', 'technical_completion_date', 'total_order_qty']].head(5))
print("\nTop Order Types:")
print(raw_df['order_type'].value_counts().head(8))
"""))

    # 06. Data Cleaning & Feature Engineering
    cells.append(nbf.v4.new_code_cell(r"""# 06. Data Cleaning & 30-Feature Engineering with Missing Value Handling
from src.preprocessing import clean_and_engineer_features, FEATURE_COLUMNS

featured_df = clean_and_engineer_features(raw_df)
print(f"Engineered Features DataFrame Shape: {featured_df.shape}")
print(f"Total Model Features: {len(FEATURE_COLUMNS)}")
print("Missing values in raw attributes are preserved as clean blanks.")
"""))

    # 07. Ground Truth Benchmark
    cells.append(nbf.v4.new_code_cell(r"""# 07. Temporal Delay Benchmark & Ground Truth Definition
delay_cnt = int(featured_df['is_delayed_exception'].sum())
delay_rate = (delay_cnt / len(featured_df)) * 100.0
print(f"Total Processed Production Orders: {len(featured_df):,}")
print(f"Ground-Truth Delayed Orders:       {delay_cnt:,} ({delay_rate:.2f}%)")
print(f"Ground-Truth On-Time Orders:       {len(featured_df) - delay_cnt:,} ({100.0 - delay_rate:.2f}%)")
"""))

    # 08. Feature Matrix & Train-Test Split
    cells.append(nbf.v4.new_code_cell(r"""# 08. Feature Matrix (X) Definition & RobustScaler Partitioning
from src.preprocessing import prepare_train_test_data

data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
X_train_scaled = data_dict['X_train_scaled']
X_test_scaled = data_dict['X_test_scaled']
y_train = data_dict['y_train']
y_test = data_dict['y_test']
scaler = data_dict['scaler']

print(f"Training Scaled Matrix X_train: {X_train_scaled.shape}")
print(f"Testing Scaled Matrix X_test:   {X_test_scaled.shape}")
print("Training is strictly UNSUPERVISED (y_train is NOT used in model fitting).")
"""))

    # 09. Suite Initialization
    cells.append(nbf.v4.new_code_cell(r"""# 09. Instantiate Unsupervised Clustering Suite
from src.models import ProductionOrderDelayClusteringSuite

contamination = max(0.10, min(0.60, delay_cnt / len(featured_df)))
suite = ProductionOrderDelayClusteringSuite(n_clusters=4, contamination=contamination, random_state=42)
print(f"Clustering Suite initialized with n_clusters=4, contamination={contamination:.4f}")
"""))

    # 10. Train Model 1: K-Means
    cells.append(nbf.v4.new_code_cell(r"""# 10. Model 1: K-Means Clustering (Centroid-Based Partitioning)
print("Training Model 1/5: K-Means Clustering...")
km = KMeans(n_clusters=4, init='k-means++', n_init=5, random_state=42)
km_labels = km.fit_predict(X_train_scaled)
suite._register_cluster_model('K-Means Clustering', km, X_train_scaled, km_labels)
print("K-Means training and cluster ordering complete.")
"""))

    # 11. Train Model 2: Gaussian Mixture
    cells.append(nbf.v4.new_code_cell(r"""# 11. Model 2: Gaussian Mixture Model (Probabilistic Density Estimation)
print("Training Model 2/5: Gaussian Mixture Model...")
gmm = GaussianMixture(n_components=4, covariance_type='diag', random_state=42, max_iter=30)
gmm_labels = gmm.fit_predict(X_train_scaled)
suite._register_cluster_model('Gaussian Mixture Model', gmm, X_train_scaled, gmm_labels)
print("Gaussian Mixture Model training and cluster ordering complete.")
"""))

    # 12. Train Model 3: Mini-Batch K-Means
    cells.append(nbf.v4.new_code_cell(r"""# 12. Model 3: Mini-Batch K-Means (Stochastic Partitioning)
print("Training Model 3/5: Mini-Batch K-Means...")
mbk = MiniBatchKMeans(n_clusters=4, batch_size=4096, n_init=5, random_state=42)
mbk_labels = mbk.fit_predict(X_train_scaled)
suite._register_cluster_model('Mini-Batch K-Means', mbk, X_train_scaled, mbk_labels)
print("Mini-Batch K-Means training complete.")
"""))

    # 13. Train Model 4: Bisecting K-Means
    cells.append(nbf.v4.new_code_cell(r"""# 13. Model 4: Bisecting K-Means (Hierarchical Partitioning)
print("Training Model 4/5: Bisecting K-Means...")
bkm = BisectingKMeans(n_clusters=4, init='k-means++', random_state=42)
bkm_labels = bkm.fit_predict(X_train_scaled)
suite._register_cluster_model('Bisecting K-Means', bkm, X_train_scaled, bkm_labels)
print("Bisecting K-Means training complete.")
"""))

    # 14. Train Model 5: PCA Latent Space Clustering
    cells.append(nbf.v4.new_code_cell(r"""# 14. Model 5: PCA Latent Space Clustering (8-Component Partitioning)
from src.models import PCALatentClusterer

print("Training Model 5/5: PCA Latent Space Clustering...")
pca_km = PCALatentClusterer(n_clusters=4, n_components=8, random_state=42)
pca_km_labels = pca_km.fit_predict(X_train_scaled)
suite._register_cluster_model('PCA Latent Clustering', pca_km, X_train_scaled, pca_km_labels)
print("PCA Latent Space Clustering training complete.")
"""))

    # 15. Predict on Held-Out Test Set
    cells.append(nbf.v4.new_code_cell(r"""# 15. Generate Test Set Predictions & Probabilities
test_preds = suite.predict_all(X_test_scaled)
print(f"Generated predictions and continuous P(Delay) scores across all {len(test_preds)} models.")
"""))

    # 16. Comprehensive Evaluation & Ranking
    cells.append(nbf.v4.new_code_cell(r"""# 16. Model Comparison Table & Ranking Selection
from src.evaluate import evaluate_models

comparison_df = evaluate_models(test_preds, y_test)
print("=" * 96)
print("5 UNSUPERVISED ML MODELS EVALUATION & RANKING TABLE")
print("=" * 96)
print(comparison_df.to_string(index=False))
print("=" * 96)

champion_model_name = comparison_df.iloc[0]['Model']
print(f"\nCHAMPION MODEL SELECTED: {champion_model_name}")
"""))

    # 17. Plot Model Comparison
    cells.append(nbf.v4.new_code_cell(r"""# 17. Performance Comparison Visualizations
from src.evaluate import plot_model_comparison

os.makedirs("reports/figures", exist_ok=True)
plot_model_comparison(comparison_df, output_dir="reports/figures")
plt.show()
"""))

    # 18. Plot ROC and PR Curves
    cells.append(nbf.v4.new_code_cell(r"""# 18. Receiver Operating Characteristic & Precision-Recall Curves
from src.evaluate import plot_roc_pr_curves

plot_roc_pr_curves(test_preds, y_test, output_dir="reports/figures")
plt.show()
"""))

    # 19. Deploy Champion Model & Score Entire Dataset
    cells.append(nbf.v4.new_code_cell(r"""# 19. Deploy #1 Champion Model & Production Scoring
from src.models import SingleModelProductionOrderSegmenter
from src.inference import score_with_single_model

champion_segmenter = SingleModelProductionOrderSegmenter(
    model_name=champion_model_name,
    n_clusters=4,
    contamination=contamination,
    random_state=42
)
champion_segmenter.fit(X_train_scaled)

scored_df = score_with_single_model(champion_segmenter, featured_df, scaler, FEATURE_COLUMNS)
print("Production scoring complete. Scored columns preview:")
print(scored_df[['order_number', 'order_type', 'plant', 'predicted_delay_probability', 'delay_prediction_label', 'delay_risk_tier', 'primary_delay_driver']].head(5))
"""))

    # 20. Feature Importance & Risk Distribution
    cells.append(nbf.v4.new_code_cell(r"""# 20. Feature Importance & Operational Delay Drivers
from src.evaluate import compute_feature_importance, plot_feature_importance, plot_delay_risk_distribution

fi_df = compute_feature_importance(X_train_scaled, y_train, FEATURE_COLUMNS)
plot_feature_importance(fi_df, output_dir="reports/figures", top_n=15)
plt.show()

plot_delay_risk_distribution(scored_df, output_dir="reports/figures")
plt.show()

print("Top 10 Delay Drivers:")
print(fi_df.head(10)[['Rank', 'Feature', 'Importance', 'Relative_Weight_Pct']].to_string(index=False))
"""))

    # 21. Multi-Sheet Excel Export & Model Persistence
    cells.append(nbf.v4.new_code_cell(r"""# 21. Multi-Sheet Executive Excel Export & Model Persistence
import shutil
from src.inference import export_enriched_excel_report, save_single_model_pipeline

output_excel = "ML_7_Production_Order_Delay_Prediction_Enriched_Output_Clean.xlsx"
export_enriched_excel_report(scored_df, output_path=output_excel, fi_df=fi_df, comp_df=comparison_df)

alt_excel = "ML_7_Production_Order_Delay_Prediction_Enriched_Output.xlsx"
try:
    shutil.copyfile(output_excel, alt_excel)
    print(f"Copied report to {alt_excel}")
except PermissionError:
    print(f"Notice: {alt_excel} is currently open in Excel. Updated file is at {output_excel}.")

save_single_model_pipeline(champion_segmenter, scaler, FEATURE_COLUMNS, output_dir="models")
print("All deliverables generated successfully!")
"""))

    nb.cells = cells

    os.makedirs("notebooks", exist_ok=True)
    nb_path_nb = os.path.join("notebooks", "07_Production_Order_Delay_Prediction_ML.ipynb")
    nb_path_root = "07_Production_Order_Delay_Prediction_ML.ipynb"

    with open(nb_path_nb, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    with open(nb_path_root, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[NOTEBOOK] Successfully generated notebook at: {nb_path_nb} and {nb_path_root}")


if __name__ == "__main__":
    create_code_only_notebook()
