"""
Generates a 100% CODE-ONLY Jupyter Notebook for Use Case 8: Quality Inspection Failure Prediction.
Every single cell is an executable Python code cell across 21 structured steps across ALL DATA.
"""
import os
import shutil
import nbformat as nbf


def create_code_only_notebook(output_path: str = "08_Quality_Inspection_Failure_Prediction_ML.ipynb"):
    nb = nbf.v4.new_notebook()
    cells = []

    # 01. Business Problem
    cells.append(nbf.v4.new_code_cell("""# 01. Business Problem & Project Overview
USE_CASE = "08 - Quality Inspection Failure Prediction (Predict Failed Inspection Lots)"
ML_PARADIGM = "Unsupervised Machine Learning (5 Models Evaluated & Ranked -> #1 Champion Selected)"
EXCEPTION_TYPE = "Quality Inspection Failure / Exception Classification"
SCOPE = "ALL DATA (HANA_P3 & HANA_P2 Without Limit)"

print("=" * 90)
print(f"USE CASE:       {USE_CASE}")
print(f"PARADIGM:       {ML_PARADIGM}")
print(f"EXCEPTION:      {EXCEPTION_TYPE}")
print(f"DATA SCOPE:     {SCOPE}")
print("OBJECTIVE:      Predict failed inspection lots from input features without labeled supervision")
print("=" * 90)
"""))

    # 02. Import Libraries
    cells.append(nbf.v4.new_code_cell("""# 02. Import Core ML & Analytics Libraries
import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)

warnings.filterwarnings('ignore')
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)
print("All core analytics and ML libraries loaded successfully.")
"""))

    # 03. Connect to Database Server
    cells.append(nbf.v4.new_code_cell("""# 03. Database Server Connection & Configuration
from src.db_connector import load_config, get_db_connection

config = load_config()
db = config["database"]
print(f"PostgreSQL Host: {db['host']}:{db['port']}")
print(f"Database Name:   {db['dbname']}")
print(f"Schemas:         {db.get('schemas', ['HANA_P3', 'HANA_P2'])}")

conn = get_db_connection(config)
print("Database connection successfully established.")
conn.close()
"""))

    # 04. Extract All Raw Quality Inspection Lots Data
    cells.append(nbf.v4.new_code_cell("""# 04. Extract All Inspection Lots from Live Tables (QALS, QAVE, MAKT, MARA, MARC, MBEW, LFA1)
from src.db_connector import extract_quality_inspection_data

raw_df = extract_quality_inspection_data(limit=None, config=config, use_cache=True)
print(f"Extracted Raw Records: {len(raw_df):,}")
print(f"Extracted Columns:     {len(raw_df.columns)}")
print(raw_df.head(3))
"""))

    # 05. Exploratory Data Inspection
    cells.append(nbf.v4.new_code_cell("""# 05. Exploratory Data Inspection & Dimensions
print("--- Inspection Type Distribution ---")
print(raw_df['inspection_type'].value_counts().head(7))

print("")
print("--- Top Manufacturing Plants ---")
print(raw_df['plant'].value_counts().head(7))

print("")
print("--- Missing Values in Raw Data (Checked Properly) ---")
for col in ['vendor_id', 'vendor_name', 'batch_number', 'material_name', 'po_number', 'ud_valuation']:
    n_null = raw_df[col].isnull().sum() + (raw_df[col].astype(str).str.strip().isin(['nan', 'None', '', '<NA>'])).sum()
    print(f"  {col}: {n_null:,} blank/missing fields ({n_null/len(raw_df)*100:.1f}%)")
"""))

    # 06. Data Cleaning & Feature Engineering
    cells.append(nbf.v4.new_code_cell("""# 06. Data Cleaning & Engineering 32 Domain Features
from src.preprocessing import clean_and_engineer_features

featured_df = clean_and_engineer_features(raw_df)
print(f"Dataset Shape after Feature Engineering: {featured_df.shape}")
print(f"Creation Date Range: {featured_df['creation_date_dt'].min().date()} to {featured_df['creation_date_dt'].max().date()}")
"""))

    # 07. Distribution of Historical Benchmark Exceptions
    cells.append(nbf.v4.new_code_cell("""# 07. Historical Benchmark Exception Ground-Truth (For Post-Hoc Evaluation Only)
fail_cnt = int(featured_df['is_failure_exception'].sum())
fail_rate = (fail_cnt / len(featured_df)) * 100.0

print(f"Total Inspection Lots:               {len(featured_df):,}")
print(f"Benchmark Failure Exception Lots:    {fail_cnt:,} ({fail_rate:.2f}%)")
print(f"Benchmark Compliant Passing Lots:   {len(featured_df) - fail_cnt:,} ({100.0 - fail_rate:.2f}%)")
"""))

    # 08. Feature Selection & Inspection Matrix Preparation
    cells.append(nbf.v4.new_code_cell("""# 08. Unsupervised 32-Feature Matrix Preparation
from src.preprocessing import get_feature_columns

feature_cols = get_feature_columns()
print(f"Total Engineered Domain Features ({len(feature_cols)}):")
for i, f in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {f}")
"""))

    # 09. Train / Test Split
    cells.append(nbf.v4.new_code_cell("""# 09. Train/Test Split (80% Train, 20% Test)
from src.preprocessing import prepare_train_test_data

data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
X_train = data_dict['X_train']
X_test = data_dict['X_test']
X_train_scaled = data_dict['X_train_scaled']
X_test_scaled = data_dict['X_test_scaled']
y_train = data_dict['y_train']
y_test = data_dict['y_test']
scaler = data_dict['scaler']

print(f"Training Matrix (Unsupervised X): {X_train_scaled.shape}")
print(f"Testing Matrix (Held-out Test):   {X_test_scaled.shape}")
print(f"Test Benchmark Exceptions:        {int(y_test.sum()):,} ({y_test.mean()*100:.2f}%)")
"""))

    # 10. RobustScaler Validation
    cells.append(nbf.v4.new_code_cell("""# 10. Validate Scaled Feature Distributions (RobustScaler)
scaled_df = pd.DataFrame(X_train_scaled[:1000], columns=feature_cols)
print("Scaled Features Check (Median ~ 0, robust to extreme outlier spikes):")
print(scaled_df[['log_lot_quantity', 'sample_to_lot_ratio', 'inspection_lead_days', 'non_conforming_ratio']].describe().T[['mean', 'std', 'min', '50%', 'max']])
"""))

    # 11. Model 1: Isolation Forest
    cells.append(nbf.v4.new_code_cell("""# 11. Train Model 1/5: Isolation Forest
from src.models import QualityInspectionFailureDetectionSuite

suite = QualityInspectionFailureDetectionSuite(contamination=0.05, random_state=42)
m1 = suite.train_isolation_forest(X_train_scaled)
print("Model 1 (Isolation Forest) trained successfully.")
"""))

    # 12. Model 2: Local Outlier Factor (LOF)
    cells.append(nbf.v4.new_code_cell("""# 12. Train Model 2/5: Local Outlier Factor (LOF)
m2 = suite.train_lof(X_train_scaled)
print("Model 2 (Local Outlier Factor) trained successfully.")
"""))

    # 13. Model 3: One-Class SVM
    cells.append(nbf.v4.new_code_cell("""# 13. Train Model 3/5: One-Class SVM
m3 = suite.train_one_class_svm(X_train_scaled)
print("Model 3 (One-Class SVM) trained successfully.")
"""))

    # 14. Model 4: DBSCAN
    cells.append(nbf.v4.new_code_cell("""# 14. Train Model 4/5: DBSCAN (Adaptive Density Clustering)
m4 = suite.train_dbscan(X_train_scaled)
print(f"Model 4 (DBSCAN) trained successfully. Adaptive eps: {m4['eps']:.4f}")
"""))

    # 15. Model 5: Deep Neural Autoencoder
    cells.append(nbf.v4.new_code_cell("""# 15. Train Model 5/5: Deep Neural Autoencoder
m5 = suite.train_autoencoder(X_train_scaled)
print(f"Model 5 (Autoencoder) trained. Reconstruction Threshold: {suite.thresholds['autoencoder']:.4f}")
"""))

    # 16. Evaluate & Rank All 5 Models
    cells.append(nbf.v4.new_code_cell("""# 16. Comprehensive Test Set Evaluation & 5-Model Accuracy Ranking Table
from src.evaluate import evaluate_models

# Evaluate on representative held-out test split
eval_X = X_test_scaled[:20000] if len(X_test_scaled) > 20000 else X_test_scaled
eval_y = y_test[:20000] if len(y_test) > 20000 else y_test

test_preds = suite.predict_all(eval_X)
comparison_df = evaluate_models(test_preds, eval_y)

print("=" * 95)
print("  5 UNSUPERVISED ML MODELS EVALUATION & ACCURACY RANKING TABLE")
print("=" * 95)
print(comparison_df.to_string(index=False))
print("=" * 95)
"""))

    # 17. Diagnostic Visualizations
    cells.append(nbf.v4.new_code_cell("""# 17. Diagnostic Visualizations: Comparative Performance Bar Chart & ROC/PR Curves
from src.evaluate import plot_model_comparison, plot_roc_pr_curves

os.makedirs("reports/figures", exist_ok=True)
plot_model_comparison(comparison_df, output_dir="reports/figures")
plot_roc_pr_curves(test_preds, eval_y, output_dir="reports/figures")
print("Model comparison and ROC/PR curve charts generated and saved in reports/figures/")
"""))

    # 18. Champion Model Selection
    cells.append(nbf.v4.new_code_cell("""# 18. Selection of #1 Ranked Champion Model
from src.models import SingleModelQualityFailureDetector

champion_model_name = comparison_df.iloc[0]['Model']
champion_acc = comparison_df.iloc[0]['Accuracy'] * 100.0
champion_f1 = comparison_df.iloc[0]['F1-Score']
champion_auc = comparison_df.iloc[0]['ROC-AUC']

print(f"SELECTED #1 CHAMPION MODEL: {champion_model_name}")
print(f"Accuracy:   {champion_acc:.2f}%")
print(f"F1-Score:   {champion_f1:.4f}")
print(f"ROC-AUC:    {champion_auc:.4f}")

champion_segmenter = SingleModelQualityFailureDetector(
    model_name=champion_model_name,
    contamination=0.05,
    random_state=42
)
champion_segmenter.fit(X_train_scaled)
print("Champion model ready for production scoring across all data.")
"""))

    # 19. Production Inference & Scoring
    cells.append(nbf.v4.new_code_cell("""# 19. Production Inference: Scoring All Inspection Lots & Calibrating P(Failure)
from src.inference import score_with_single_model
from src.evaluate import plot_failure_risk_distribution

scored_df = score_with_single_model(champion_segmenter, featured_df, scaler, feature_cols)
plot_failure_risk_distribution(scored_df, output_dir="reports/figures")

print("")
print("--- Quality Risk Tier Triage Breakdown (All Data) ---")
print(scored_df['risk_category'].value_counts())
"""))

    # 20. Feature Importance
    cells.append(nbf.v4.new_code_cell("""# 20. Feature Importance & Domain Impact Explainability
from src.evaluate import compute_feature_importance, plot_feature_importance

fi_df = compute_feature_importance(champion_segmenter.model, feature_cols, X_train_scaled, y_train)
fi_df.to_csv("reports/feature_importance.csv", index=False)
plot_feature_importance(fi_df, output_dir="reports/figures", top_n=15)

print("")
print("Top 10 Influential Quality Failure Drivers:")
print(fi_df.head(10)[['Rank', 'Feature', 'Importance', 'Relative_Weight_Pct']].to_string(index=False))
"""))

    # 21. Multi-Sheet Excel Export & Persistence
    cells.append(nbf.v4.new_code_cell("""# 21. Multi-Sheet Executive Excel Deliverable Export & Artifact Persistence
import shutil
from src.inference import export_enriched_excel_report, save_single_model_pipeline

output_excel = "ML_8_Quality_Inspection_Failure_Prediction_Enriched_Output_Clean.xlsx"
export_enriched_excel_report(scored_df, output_path=output_excel, fi_df=fi_df, comp_df=comparison_df)

alt_excel = "ML_8_Quality_Inspection_Failure_Prediction_Enriched_Output.xlsx"
shutil.copyfile(output_excel, alt_excel)
shutil.copyfile(output_excel, os.path.join("reports", os.path.basename(output_excel)))
shutil.copyfile(output_excel, os.path.join("reports", alt_excel))

save_single_model_pipeline(champion_segmenter, scaler, feature_cols, output_dir="models")
print("All deliverables and artifacts successfully exported and saved.")
"""))

    nb.cells = cells
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[NOTEBOOK] Successfully created 100% code-only notebook at: {output_path}")

    # Also save in notebooks/
    nb_dir_path = os.path.join("notebooks", os.path.basename(output_path))
    os.makedirs("notebooks", exist_ok=True)
    with open(nb_dir_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[NOTEBOOK] Copied notebook to: {nb_dir_path}")


if __name__ == "__main__":
    create_code_only_notebook()
