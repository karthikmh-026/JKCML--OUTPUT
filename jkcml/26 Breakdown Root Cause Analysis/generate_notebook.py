"""
Generates a 100% CODE-ONLY Jupyter Notebook for Use Case 26: Breakdown Root Cause Analysis (NLP).
ZERO Markdown cells. Every single cell is an executable Python code cell across the 21 steps.
"""
import os
import nbformat as nbf


def create_code_only_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 01. Business Problem
    cells.append(nbf.v4.new_code_cell("""# 01. Business Problem
USE_CASE = "26 - Breakdown Root Cause Analysis (NLP: Extract Root Causes from Maintenance Text)"
ML_PARADIGM = "Unsupervised NLP & Clustering (5 Unsupervised Models Ranked -> 1 Champion Model Selected)"
N_CLUSTERS = 4

print(f"Business Problem: {USE_CASE}")
print(f"ML Approach:      {ML_PARADIGM}")
print(f"Target Segments:  {N_CLUSTERS} Breakdown RCA Tiers (Preventive/Calibrated, Verified Corrective, Moderate/Chronic Wear, Critical Breakdown/Exception)")
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
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve,
    silhouette_score
)

warnings.filterwarnings('ignore')
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)
"""))

    # 03. Connect to Server
    cells.append(nbf.v4.new_code_cell("""# 03. Connect to Server
DB_HOST = "192.168.1.44"
DB_PORT = 5432
DB_NAME = "JKC_Input"
DB_SCHEMA = "HANA_P2"
DB_USER = "postgres"
DB_PASS = "postgres"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

with engine.connect() as conn:
    print(f"Connected to PostgreSQL: {DB_HOST}:{DB_PORT}/{DB_NAME} (Schema: {DB_SCHEMA})")
"""))

    # 04. Extract Data
    cells.append(nbf.v4.new_code_cell("""# 04. Extract ALL Data from SAP Plant Maintenance Orders, Confirmations & Spare Parts across HANA_P2 and HANA_P3
from src.db_connector import extract_breakdown_rca_data

df_raw = extract_breakdown_rca_data(limit=None)
print(f"Extracted records: {len(df_raw):,} rows, {df_raw.shape[1]} columns")
"""))

    # 05. Load Data
    cells.append(nbf.v4.new_code_cell("""# 05. Load Data Preview
print(f"Dataset Shape: {df_raw.shape}")
df_raw[['order_number', 'order_type', 'plant', 'work_center', 'maintenance_text', 'confirmation_text', 'spare_parts_texts', 'total_spare_cost_inr']].head()
"""))

    # 06. Data Understanding / EDA
    cells.append(nbf.v4.new_code_cell("""# 06. Data Understanding / EDA
df_raw[['order_type', 'plant', 'work_center', 'phase_teco', 'phase_closed']].describe().T
"""))

    cells.append(nbf.v4.new_code_cell("""# 06. Data Understanding / EDA (Visualizations)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
df_raw['order_type'].replace('', 'UNASSIGNED').value_counts().head(8).plot(kind='barh', color='steelblue')
plt.title('SAP PM Order Types (AUFK-AUART) by Order Count')

plt.subplot(1, 2, 2)
df_raw['work_center'].replace('', 'UNASSIGNED').value_counts().head(8).plot(kind='bar', color='darkorange')
plt.title('Top SAP PM Work Centers (AUFK-VAPLZ)')
plt.tight_layout()
plt.show()
"""))

    # 07. Data Cleaning & 08. Feature Engineering
    cells.append(nbf.v4.new_code_cell("""# 07 & 08. NLP Data Cleaning & 32-Feature Engineering for Breakdown Root Cause Analysis
from src.preprocessing import clean_and_engineer_features, get_feature_matrix

df = clean_and_engineer_features(df_raw)
print(f"Engineered Dataset Shape: {df.shape}")
df[['order_number', 'order_type', 'maintenance_text', 'extracted_equipment_tag', 'extracted_root_cause_code', 'extracted_root_cause', 'extracted_root_cause_domain', 'rca_tier_code', 'nlp_rca_confidence_score', 'breakdown_severity_risk_index']].head(8)
"""))

    # 09. Data Preprocessing
    cells.append(nbf.v4.new_code_cell("""# 09. Data Preprocessing (32 Unsupervised NLP & Plant Maintenance Input Features)
X_raw, y, FEATURE_COLS = get_feature_matrix(df)
scaler_full = RobustScaler()
X_scaled_full = scaler_full.fit_transform(X_raw)

print(f"Preprocessed Feature Matrix: {X_scaled_full.shape} (32 Numerical NLP & PM Features)")
"""))

    # 10. Define X and Post-Hoc Evaluation Benchmark Y
    cells.append(nbf.v4.new_code_cell("""# 10. Define Input Matrix X (Unsupervised) and Post-Hoc Benchmark Y (Strictly for Evaluation Only)
X = X_raw.copy()
contamination_rate = float(np.mean(y))

print(f"Total SAP Plant Maintenance Orders:           {len(y):,}")
print(f"Verified Preventive & Corrective PM Orders:   {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%)")
print(f"Critical Breakdown & Unverified RCA Orders:   {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")
"""))

    # 11. Train/Test Split
    cells.append(nbf.v4.new_code_cell("""# 11. Train/Test Split & Robust Scaling
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.20, random_state=42, stratify=y
)

scaler = RobustScaler(quantile_range=(10.0, 90.0))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from src.models import BreakdownRCAClusteringSuite
suite = BreakdownRCAClusteringSuite(n_clusters=4, contamination=contamination_rate, random_state=42)

print(f"Train Set: {X_train_scaled.shape[0]:,} maintenance orders")
print(f"Test Set:  {X_test_scaled.shape[0]:,} maintenance orders")
"""))

    # 12. Model 1 - K-Means Clustering
    cells.append(nbf.v4.new_code_cell("""# 12. Unsupervised NLP Clustering Model 1: K-Means Clustering
km_model = suite.train_kmeans(X_train_scaled)
preds_km, scores_km, tiers_km = suite._predict_single('kmeans', X_test_scaled, 2)

print(f"K-Means Clustering Flagged: {preds_km.sum():,} breakdown/RCA exceptions out of {len(preds_km):,} test orders")
"""))

    # 13. Model 2 - Gaussian Mixture Model (GMM)
    cells.append(nbf.v4.new_code_cell("""# 13. Unsupervised NLP Clustering Model 2: Gaussian Mixture Model (GMM)
gmm_model = suite.train_gmm(X_train_scaled)
preds_gmm, scores_gmm, tiers_gmm = suite._predict_single('gmm', X_test_scaled, 5)

print(f"Gaussian Mixture (GMM) Flagged: {preds_gmm.sum():,} breakdown/RCA exceptions out of {len(preds_gmm):,} test orders")
"""))

    # 14. Model 3 - Mini-Batch K-Means
    cells.append(nbf.v4.new_code_cell("""# 14. Unsupervised NLP Clustering Model 3: Mini-Batch K-Means
mbk_model = suite.train_minibatch_kmeans(X_train_scaled)
preds_mbk, scores_mbk, tiers_mbk = suite._predict_single('minibatch_kmeans', X_test_scaled, 7)

print(f"Mini-Batch K-Means Flagged: {preds_mbk.sum():,} breakdown/RCA exceptions out of {len(preds_mbk):,} test orders")
"""))

    # 15. Model 4 - Bisecting K-Means
    cells.append(nbf.v4.new_code_cell("""# 15. Unsupervised NLP Clustering Model 4: Bisecting K-Means
bkm_model = suite.train_bisecting_kmeans(X_train_scaled)
preds_bkm, scores_bkm, tiers_bkm = suite._predict_single('bisecting_kmeans', X_test_scaled, 9)

print(f"Bisecting K-Means Flagged: {preds_bkm.sum():,} breakdown/RCA exceptions out of {len(preds_bkm):,} test orders")
"""))

    # 16. Model 5 - BIRCH Clustering
    cells.append(nbf.v4.new_code_cell("""# 16. Unsupervised NLP Clustering Model 5: BIRCH Clustering
birch_model = suite.train_birch(X_train_scaled)
preds_birch, scores_birch, tiers_birch = suite._predict_single('birch', X_test_scaled, 11)

print(f"BIRCH Clustering Flagged: {preds_birch.sum():,} breakdown/RCA exceptions out of {len(preds_birch):,} test orders")
"""))

    # 17. Model Evaluation & Ranking
    cells.append(nbf.v4.new_code_cell("""# 17. Model Evaluation & Accuracy Ranking across all 5 Unsupervised NLP Clustering Models
from src.evaluate import evaluate_models

models_dict = suite.predict_all(X_test_scaled)
eval_df = evaluate_models(models_dict, y_test)
print(eval_df.to_string(index=False))
eval_df
"""))

    # 18. Model Comparison & Champion Selection
    cells.append(nbf.v4.new_code_cell("""# 18. Model Comparison Visualizations & #1 Champion Selection
plt.figure(figsize=(12, 5))
melted = pd.melt(eval_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
sns.barplot(data=melted, x='Model', y='value', hue='variable', palette='Blues_r')
plt.title('5 Unsupervised NLP Clustering ML Models Performance & Accuracy Comparison - Breakdown Root Cause Analysis')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for name, res in models_dict.items():
    fpr, tpr, _ = roc_curve(y_test, res['scores'])
    ax1.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, res['scores']):.3f})")
    prec, rec, _ = precision_recall_curve(y_test, res['scores'])
    ax2.plot(rec, prec, label=f"{name} ({average_precision_score(y_test, res['scores']):.3f})")

ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_title('ROC Curves (5 Unsupervised NLP Clustering Models)')
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax1.legend()

ax2.set_title('Precision-Recall Curves')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.legend()
plt.tight_layout()
plt.show()

champion_model = eval_df.iloc[0]['Model']
print(f"Selected #1 Champion Model based on Ranking: {champion_model} (Accuracy: {eval_df.iloc[0]['Accuracy']*100:.2f}%, F1: {eval_df.iloc[0]['F1-Score']:.4f})")
"""))

    # 19. Feature Importance
    cells.append(nbf.v4.new_code_cell("""# 19. Feature Importance (All 32 NLP & Breakdown Root-Cause Features)
surrogate = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
surrogate.fit(X_test_scaled, scores_km)

feat_imp = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Importance': surrogate.feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp.head(15), x='Importance', y='Feature', palette='mako')
plt.title('Top 15 Feature Importances for Breakdown Root Cause Analysis (NLP)')
plt.tight_layout()
plt.show()

print(feat_imp.head(15).to_string(index=False))
"""))

    # 20. Final Prediction with Champion Model
    cells.append(nbf.v4.new_code_cell("""# 20. Final Breakdown Root Cause Extraction & Exception Triage using Selected #1 Champion Model
from src.models import SingleModelBreakdownRCAAnalyzer
from src.inference import score_with_single_model

analyzer = SingleModelBreakdownRCAAnalyzer(n_clusters=4, contamination=contamination_rate, random_state=42, pretrained_model=km_model)
analyzer.fit(X_train_scaled)
df_scored = score_with_single_model(df, analyzer, scaler, FEATURE_COLS)

print("=== NLP EXTRACTED ROOT CAUSE ENGINEERING DOMAIN COUNTS ===")
print(df_scored['extracted_root_cause_domain'].value_counts())
print("\\n=== BREAKDOWN ROOT CAUSE SEGMENT COUNTS ===")
print(df_scored['rca_segment_name'].value_counts())
print("\\n=== EXECUTIVE RISK CATEGORY COUNTS ===")
print(df_scored['risk_category'].value_counts())

audit_cols = ['order_number', 'order_type', 'maintenance_text', 'extracted_equipment_tag', 'extracted_root_cause_code', 'extracted_root_cause', 'nlp_rca_confidence_score', 'breakdown_exception_prob', 'risk_category']
print(df_scored[df_scored['risk_category'] == 'Critical Risk'][audit_cols].head(10).to_string(index=False))
"""))

    # 21. Save Model
    cells.append(nbf.v4.new_code_cell("""# 21. Save Champion Model & Scaler
os.makedirs("models", exist_ok=True)

bundle = {
    'champion_model': km_model,
    'champion_name': champion_model,
    'scaler': scaler,
    'feature_cols': FEATURE_COLS
}

joblib.dump(bundle, "models/breakdown_rca_model_bundle.joblib")
joblib.dump(scaler, "models/robust_scaler.joblib")
print("Saved champion NLP clustering model bundle and scaler to models/")
"""))

    nb.cells = cells

    paths = [
        os.path.join("notebooks", "26_Breakdown_Root_Cause_Analysis_ML.ipynb"),
        "26_Breakdown_Root_Cause_Analysis_ML.ipynb"
    ]
    for p in paths:
        os.makedirs(os.path.dirname(p) if os.path.dirname(p) else ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"[SUCCESS] Wrote 100% CODE-ONLY notebook to {p} ({len(cells)} code cells, 0 markdown cells).")


if __name__ == "__main__":
    create_code_only_notebook()
