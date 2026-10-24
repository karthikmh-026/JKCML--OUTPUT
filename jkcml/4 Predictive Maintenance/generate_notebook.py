"""
Generates a comprehensive executable Jupyter Notebook for Use Case 4:
Predictive Maintenance Classification (Predict Equipment Failures Before Breakdown)
Covers all 21 modular ML steps:
01. Business Problem
02. Import Libraries
03. Connect to Database (PostgreSQL 192.168.1.44:5432 / HANA_P2)
04. Extract SAP Plant Maintenance Data (AUFK, AFKO, AFRU, AUFM, CRHD, T001W, T001)
05. Exploratory Data Analysis (EDA)
06. Data Cleaning & Handling Missing Values
07. 32-Feature Engineering (Operating Stress, Confirmations, Spare Costs, NLP Wear Cues)
08. Post-Hoc Audit Validation Benchmark Definition
09. Unsupervised Train/Test Splitting (Strictly Input Features X)
10. Robust Scaling
11. Model 1 Training: Isolation Forest
12. Model 2 Training: Local Outlier Factor (LOF)
13. Model 3 Training: One-Class SVM
14. Model 4 Training: Gaussian Mixture Model (GMM)
15. Model 5 Training: K-Means Clustering
16. Accuracy & Evaluation Metrics Calculation (5 Models Comparison Table)
17. Visualizations: Model Comparison & ROC/PR Curves
18. Objective Ranking & Selection of #1 Champion Model
19. 32-Feature Importance & Explainability
20. Scoring Full Dataset: Health Tiers & Prescriptive Maintenance Actions
21. Export Executive Multi-Tab Excel Reports & Model Artifacts
"""
import os
import nbformat as nbf


def create_predictive_maintenance_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 01. Business Problem
    cells.append(nbf.v4.new_code_cell("""# 01. Business Problem
USE_CASE = "4 - Predictive Maintenance Classification (Predict Equipment Failures Before Breakdown)"
ML_PARADIGM = "100% Unsupervised Machine Learning (5 ML Models Evaluated & Ranked -> #1 Champion Selected)"
EXCEPTION_TYPE = "Equipment Failure & Critical Breakdown Risk Exception"

print(f"Business Problem: {USE_CASE}")
print(f"ML Paradigm:      {ML_PARADIGM}")
print(f"Exception Focus:  {EXCEPTION_TYPE}")
print("Objective:        Identify equipment degradation patterns, spare part surges, and operating stresses to prevent breakdown.")
"""))

    # 02. Import Libraries
    cells.append(nbf.v4.new_code_cell("""# 02. Import Libraries
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)

warnings.filterwarnings('ignore')
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)
print("All core libraries imported successfully.")
"""))

    # 03. Connect to Database
    cells.append(nbf.v4.new_code_cell("""# 03. Connect to Database
DB_HOST = "192.168.1.44"
DB_PORT = 5432
DB_NAME = "JKC_Input"
DB_SCHEMA = "HANA_P2"
DB_USER = "postgres"
DB_PASS = "postgres"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
with engine.connect() as conn:
    print(f"Connected successfully to PostgreSQL server at {DB_HOST}:{DB_PORT}, Database: {DB_NAME}, Schema: {DB_SCHEMA}")
"""))

    # 04. Extract SAP Plant Maintenance Data
    cells.append(nbf.v4.new_code_cell("""# 04. Extract SAP Plant Maintenance Data
from src.db_connector import extract_predictive_maintenance_data

print("Extracting joined PM orders, confirmations, spare parts, and work centers from HANA_P2 and HANA_P3 (Full Dataset without Limits)...")
raw_df = extract_predictive_maintenance_data(limit=None)
print(f"Total Extracted Maintenance Records: {len(raw_df):,}")
print("Order Types Distribution:")
print(raw_df['order_type'].value_counts())
"""))

    # 05. Exploratory Data Analysis
    cells.append(nbf.v4.new_code_cell("""# 05. Exploratory Data Analysis (EDA)
print("Data Shape:", raw_df.shape)
print("Sample Columns:", raw_df.columns.tolist()[:15])
print("")
print("Numerical Statistics (Confirmations, Duration, Spare Parts Cost):")
print(raw_df[['confirmation_count', 'total_actual_duration', 'total_labor_hours', 'spare_parts_item_count', 'total_spare_parts_cost']].describe())
"""))

    # 06. Data Cleaning & Handling Missing Values
    cells.append(nbf.v4.new_code_cell("""# 06. Data Cleaning & Handling Missing Values
null_counts = raw_df[['order_number', 'order_type', 'plant_code', 'work_center']].isnull().sum()
print("Critical Columns Null Count:")
print(null_counts)
"""))

    # 07. 32-Feature Engineering
    cells.append(nbf.v4.new_code_cell("""# 07. 32-Feature Engineering
from src.preprocessing import clean_and_engineer_features, FEATURE_COLUMNS

featured_df = clean_and_engineer_features(raw_df)
print(f"Engineered Dataset Shape: {featured_df.shape}")
print(f"Total Features ({len(FEATURE_COLUMNS)}):")
for i, feat in enumerate(FEATURE_COLUMNS):
    print(f"  {i+1:02d}. {feat}")
"""))

    # 08. Post-Hoc Audit Validation Benchmark Definition
    cells.append(nbf.v4.new_code_cell("""# 08. Post-Hoc Audit Validation Benchmark Definition
benchmark_cnt = int(featured_df['is_equipment_failure_exception'].sum())
benchmark_rate = (benchmark_cnt / len(featured_df)) * 100.0
print(f"Historical Breakdown & High-Stress Failure Benchmark Exceptions: {benchmark_cnt:,} ({benchmark_rate:.2f}%)")
print("Note: This label is held-out purely for objective validation and ranking of the 5 unsupervised models.")
"""))

    # 09. Unsupervised Train/Test Splitting
    cells.append(nbf.v4.new_code_cell("""# 09. Unsupervised Train/Test Splitting (Strictly Input Features X)
from src.preprocessing import prepare_train_test_data

data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
X_train_scaled = data_dict['X_train_scaled']
X_test_scaled = data_dict['X_test_scaled']
X_test_df = data_dict['X_test_df']
y_train = data_dict['y_train']
y_test = data_dict['y_test']
scaler = data_dict['scaler']

print(f"Training Matrix X_train_scaled: {X_train_scaled.shape} (100% Unsupervised, No Target Labels)")
print(f"Testing Matrix  X_test_scaled:  {X_test_scaled.shape}")
print(f"Held-out Validation Labels:     {len(y_test):,} samples")
"""))

    # 10. Robust Scaling
    cells.append(nbf.v4.new_code_cell("""# 10. Robust Scaling
print("RobustScaler median values (first 5 features):", scaler.center_[:5])
print("RobustScaler scale values  (first 5 features):", scaler.scale_[:5])
"""))

    # 11-15. Train 5 Unsupervised Models
    cells.append(nbf.v4.new_code_cell("""# 11-15. Train 5 Unsupervised Machine Learning Models
from src.models import PredictiveMaintenanceUnsupervisedSuite

contamination = max(0.05, min(0.40, benchmark_cnt / len(featured_df)))
suite = PredictiveMaintenanceUnsupervisedSuite(contamination=contamination, random_state=42)

print("Training Model 1: Isolation Forest...")
suite.train_isolation_forest(X_train_scaled)

print("Training Model 2: Local Outlier Factor (LOF)...")
suite.train_lof(X_train_scaled)

print("Training Model 3: One-Class SVM...")
suite.train_one_class_svm(X_train_scaled)

print("Training Model 4: Gaussian Mixture Model (GMM)...")
suite.train_gmm(X_train_scaled)

print("Training Model 5: K-Means Clustering...")
suite.train_kmeans(X_train_scaled)

print("All 5 Unsupervised Machine Learning Models trained successfully.")
"""))

    # 16. Accuracy & Evaluation Metrics Calculation (5 Models Comparison Table)
    cells.append(nbf.v4.new_code_cell("""# 16. Accuracy & Evaluation Metrics Calculation (5 Models Comparison Table)
from src.evaluate import evaluate_models

test_preds = suite.predict_all(X_test_scaled)
comparison_df = evaluate_models(test_preds, y_test)

print("=" * 96)
print("5 UNSUPERVISED ML MODELS ACCURACY & PERFORMANCE RANKING TABLE")
print("=" * 96)
print(comparison_df.to_string(index=False))
print("=" * 96)
"""))

    # 17. Visualizations: Model Comparison & ROC/PR Curves
    cells.append(nbf.v4.new_code_cell("""# 17. Visualizations: Model Comparison & ROC/PR Curves
from src.evaluate import plot_model_comparison, plot_roc_pr_curves, plot_confusion_matrix

os.makedirs("reports/figures", exist_ok=True)
plot_model_comparison(comparison_df, output_dir="reports/figures")
plot_roc_pr_curves(test_preds, y_test, output_dir="reports/figures")

champion_name = comparison_df.iloc[0]['Model']
champion_key = comparison_df.iloc[0]['Model_Key']
plot_confusion_matrix(y_test, test_preds[champion_key]['preds'], champion_name=champion_name, output_dir="reports/figures")

print("Generated plots: model_comparison_metrics.png, roc_pr_curves.png, confusion_matrix.png")
"""))

    # 18. Objective Ranking & Selection of #1 Champion Model
    cells.append(nbf.v4.new_code_cell("""# 18. Objective Ranking & Selection of #1 Champion Model
from src.models import SingleModelPredictiveMaintenanceClassifier

champion_row = comparison_df.iloc[0]
print(f"Rank #1 Champion Model Selected: {champion_row['Model']} ({champion_row['Model_Key']})")
print(f"  -> Accuracy:   {champion_row['Accuracy']*100:.2f}%")
print(f"  -> Precision:  {champion_row['Precision']*100:.2f}%")
print(f"  -> Recall:     {champion_row['Recall']*100:.2f}%")
print(f"  -> F1-Score:   {champion_row['F1-Score']:.4f}")
print(f"  -> ROC-AUC:    {champion_row['ROC-AUC']:.4f}")

champion_classifier = SingleModelPredictiveMaintenanceClassifier(
    model_key=champion_row['Model_Key'],
    model_instance=suite.models[champion_row['Model_Key']],
    scaler=scaler,
    threshold=suite.thresholds[champion_row['Model_Key']],
    meta=suite.meta[champion_row['Model_Key']],
    feature_cols=FEATURE_COLUMNS
)
"""))

    # 19. 32-Feature Importance & Explainability
    cells.append(nbf.v4.new_code_cell("""# 19. 32-Feature Importance & Explainability
from src.evaluate import compute_feature_importance

fi_df = compute_feature_importance(champion_classifier, X_test_df, FEATURE_COLUMNS)
print("")
print("Top 10 Most Important Features Driving Equipment Failure Predictions:")
print(fi_df.head(10).to_string(index=False))
"""))

    # 20. Scoring Full Dataset: Health Tiers & Prescriptive Maintenance Actions
    cells.append(nbf.v4.new_code_cell("""# 20. Scoring Full Dataset: Health Tiers & Prescriptive Maintenance Actions
from src.inference import score_with_single_model

enriched_df = score_with_single_model(champion_classifier, featured_df)
print("Sample Enriched Maintenance Predictions:")
print(enriched_df[['order_number', 'order_type', 'plant_name', 'equipment_health_tier', 'failure_risk_score', 'prescriptive_action']].head(5).to_string())

print("")
print("Equipment Health Tiers Distribution:")
print(enriched_df['equipment_health_tier'].value_counts())
"""))

    # 21. Export Executive Multi-Tab Excel Reports & Model Artifacts
    cells.append(nbf.v4.new_code_cell("""# 21. Export Executive Multi-Tab Excel Reports & Model Artifacts
from src.inference import export_enriched_excel_report, save_single_model_pipeline

save_single_model_pipeline(
    champion_classifier,
    model_path="models/champion_predictive_maintenance_model.joblib",
    scaler_path="models/predictive_maintenance_scaler.joblib"
)

export_enriched_excel_report(
    enriched_df, comparison_df, fi_df,
    output_excel_path="ML_4_Predictive_Maintenance_Enriched_Output_Clean.xlsx"
)
export_enriched_excel_report(
    enriched_df, comparison_df, fi_df,
    output_excel_path="ML_4_Predictive_Maintenance_Enriched_Output.xlsx"
)

print("Exported multi-sheet Excel reports and serialized model artifacts.")
print("Predictive Maintenance pipeline completed.")
"""))

    nb['cells'] = cells
    os.makedirs("notebooks", exist_ok=True)
    out_path = "notebooks/04_Predictive_Maintenance_ML.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[INFO] Jupyter Notebook generated successfully: {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    create_predictive_maintenance_notebook()
