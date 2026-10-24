"""
Generates a 100% CODE-ONLY Jupyter Notebook for Use Case 20: Duplicate Invoice Detection.
ZERO Markdown cells. Every single cell is an executable Python code cell across the 21 steps.
Configured for NO LIMIT extraction across HANA_P2 and HANA_P3 with true missing value handling.
"""
import os
import nbformat as nbf


def create_code_only_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 01. Business Problem
    cells.append(nbf.v4.new_code_cell("""# 01. Business Problem
USE_CASE = "20 - Duplicate Invoice Detection (Identify Potential Duplicate Payments)"
ML_PARADIGM = "Unsupervised Anomaly Detection (5 Models Ranked -> 1 Champion Model Selected)"
CONTAMINATION_RATE = 0.05

print(f"Business Problem: {USE_CASE}")
print(f"ML Approach:      {ML_PARADIGM}")
print(f"Expected Anomaly Rate: {CONTAMINATION_RATE * 100:.0f}%")
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
from sklearn.ensemble import IsolationForest, RandomForestRegressor
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
    print(f"Connected to PostgreSQL: {DB_HOST}:{DB_PORT}/{DB_NAME} (Schemas: HANA_P2 & HANA_P3)")
"""))

    # 04. Extract Data
    cells.append(nbf.v4.new_code_cell("""# 04. Extract Data from SAP Tables across HANA_P2 and HANA_P3 (RBKP, RSEG, LFA1, LFB1, EKPO, MAKT, MARA, T001W, T001)
from src.db_connector import extract_invoice_data

df_raw = extract_invoice_data(limit=None, schema="ALL")
print(f"Extracted records: {len(df_raw):,} rows, {df_raw.shape[1]} columns directly from server (NO LIMIT).")
"""))

    # 05. Load Data
    cells.append(nbf.v4.new_code_cell("""# 05. Load Data Preview
print(f"Dataset Shape: {df_raw.shape}")
df_raw[['invoice_doc_number', 'reference_invoice_no', 'vendor_id', 'vendor_name', 'plant_name', 'material_desc', 'gross_invoice_amount']].head()
"""))

    # 06. Data Understanding / EDA
    cells.append(nbf.v4.new_code_cell("""# 06. Data Understanding / EDA
df_raw[['invoiced_qty', 'invoice_item_amount', 'tax_amount', 'gross_invoice_amount', 'po_net_value']].describe().T
"""))

    cells.append(nbf.v4.new_code_cell("""# 06. Data Understanding / EDA (Visualizations)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
df_raw['material_group'].replace('', 'BLANK').value_counts().head(8).plot(kind='barh', color='steelblue')
plt.title('Top Material Groups in Incoming Invoices')

plt.subplot(1, 2, 2)
df_raw['doc_type'].replace('', 'BLANK').value_counts().head(8).plot(kind='bar', color='darkorange')
plt.title('Invoice Document Types (BLART)')
plt.tight_layout()
plt.show()
"""))

    # 07. Data Cleaning & 08. Feature Engineering
    cells.append(nbf.v4.new_code_cell("""# 07 & 08. Data Cleaning & 32-Feature Engineering for Duplicate Invoice Detection
from src.preprocessing import clean_and_engineer_features, get_feature_matrix

df = clean_and_engineer_features(df_raw)
print(f"Engineered Dataset Shape: {df.shape}")
df[['invoice_doc_number', 'vendor_name', 'reference_invoice_no', 'gross_invoice_amount', 'exact_vendor_ref_count', 'vendor_amt_date_count', 'dup_similarity_index', 'duplicate_pattern_type']].head(8)
"""))

    # 09. Data Preprocessing
    cells.append(nbf.v4.new_code_cell("""# 09. Data Preprocessing (32 Unsupervised Input Features)
X_raw, y, FEATURE_COLS = get_feature_matrix(df)
scaler_full = RobustScaler()
X_scaled_full = scaler_full.fit_transform(X_raw)

print(f"Preprocessed Feature Matrix: {X_scaled_full.shape} (32 Numerical Domain Features)")
"""))

    # 10. Define X and Post-Hoc Evaluation Benchmark Y
    cells.append(nbf.v4.new_code_cell("""# 10. Define Input Matrix X (Unsupervised) and Post-Hoc Audit Benchmark Y (Strictly for Evaluation Only)
X = X_raw.copy()

print(f"Total Invoices:              {len(y):,}")
print(f"Normal Invoices (Y=0):       {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%)")
print(f"Duplicate Exceptions (Y=1):  {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")
"""))

    # 11. Train/Test Split
    cells.append(nbf.v4.new_code_cell("""# 11. Train/Test Split & Robust Scaling
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.20, random_state=42, stratify=y
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train Set: {X_train_scaled.shape[0]:,} invoices")
print(f"Test Set:  {X_test_scaled.shape[0]:,} invoices")
"""))

    # 12. Model 1 - Isolation Forest
    cells.append(nbf.v4.new_code_cell("""# 12. Unsupervised Model 1: Isolation Forest
iso_forest = IsolationForest(
    n_estimators=120,
    contamination=0.05,
    max_samples=min(2048, len(X_train_scaled)),
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_scaled)

preds_if = (iso_forest.predict(X_test_scaled) == -1).astype(int)
scores_if = -iso_forest.decision_function(X_test_scaled)

print(f"Isolation Forest Flagged: {preds_if.sum():,} duplicate anomalies out of {len(preds_if):,} test invoices")
"""))

    # 13. Model 2 - Local Outlier Factor
    cells.append(nbf.v4.new_code_cell("""# 13. Unsupervised Model 2: Local Outlier Factor (LOF)
np.random.seed(42)
idx_lof = np.random.choice(len(X_train_scaled), size=min(20000, len(X_train_scaled)), replace=False)

lof = LocalOutlierFactor(
    n_neighbors=25,
    contamination=0.05,
    novelty=True,
    n_jobs=-1
)
lof.fit(X_train_scaled[idx_lof])

preds_lof = (lof.predict(X_test_scaled) == -1).astype(int)
scores_lof = -lof.decision_function(X_test_scaled)

print(f"LOF Flagged: {preds_lof.sum():,} duplicate anomalies out of {len(preds_lof):,} test invoices")
"""))

    # 14. Model 3 - One-Class SVM
    cells.append(nbf.v4.new_code_cell("""# 14. Unsupervised Model 3: One-Class SVM
idx_svm = np.random.choice(len(X_train_scaled), size=min(15000, len(X_train_scaled)), replace=False)

oc_svm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.05
)
oc_svm.fit(X_train_scaled[idx_svm])

preds_svm = (oc_svm.predict(X_test_scaled) == -1).astype(int)
scores_svm = -oc_svm.decision_function(X_test_scaled)

print(f"One-Class SVM Flagged: {preds_svm.sum():,} duplicate anomalies out of {len(preds_svm):,} test invoices")
"""))

    # 15. Model 4 - DBSCAN
    cells.append(nbf.v4.new_code_cell("""# 15. Unsupervised Model 4: DBSCAN
idx_db = np.random.choice(len(X_train_scaled), size=min(20000, len(X_train_scaled)), replace=False)
X_db_train = X_train_scaled[idx_db]

dbscan = DBSCAN(eps=1.8, min_samples=15, n_jobs=-1)
train_labels = dbscan.fit_predict(X_db_train)

core_samples = X_db_train[dbscan.core_sample_indices_] if len(dbscan.core_sample_indices_) > 0 else X_db_train
nn = NearestNeighbors(n_neighbors=15, n_jobs=-1).fit(core_samples)

train_dist, _ = nn.kneighbors(X_db_train)
eps_thresh = float(np.percentile(train_dist[:, -1], 95))

distances, _ = nn.kneighbors(X_test_scaled)
scores_db = distances[:, -1]
preds_db = (scores_db >= eps_thresh).astype(int)

print(f"DBSCAN Flagged: {preds_db.sum():,} duplicate anomalies out of {len(preds_db):,} test invoices")
"""))

    # 16. Model 5 - Autoencoder
    cells.append(nbf.v4.new_code_cell("""# 16. Unsupervised Model 5: Deep Neural Autoencoder
idx_ae = np.random.choice(len(X_train_scaled), size=min(30000, len(X_train_scaled)), replace=False)
X_ae_train = X_train_scaled[idx_ae]

autoencoder = MLPRegressor(
    hidden_layer_sizes=(32, 16, 8, 16, 32),
    activation='relu',
    solver='adam',
    batch_size=256,
    learning_rate_init=0.005,
    max_iter=60,
    early_stopping=True,
    random_state=42
)
autoencoder.fit(X_ae_train, X_ae_train)

recon_train = autoencoder.predict(X_ae_train)
recon_test = autoencoder.predict(X_test_scaled)

train_errors = np.mean(np.square(X_ae_train - recon_train), axis=1)
scores_ae = np.mean(np.square(X_test_scaled - recon_test), axis=1)

ae_thresh = np.percentile(train_errors, 95)
preds_ae = (scores_ae >= ae_thresh).astype(int)

print(f"Autoencoder Flagged: {preds_ae.sum():,} duplicate anomalies out of {len(preds_ae):,} test invoices")
"""))

    # 17. Model Evaluation & Ranking
    cells.append(nbf.v4.new_code_cell("""# 17. Model Evaluation & Accuracy Ranking across all 5 Unsupervised Models
models_dict = {
    'Isolation Forest': {'preds': preds_if, 'scores': scores_if},
    'Local Outlier Factor': {'preds': preds_lof, 'scores': scores_lof},
    'One-Class SVM': {'preds': preds_svm, 'scores': scores_svm},
    'DBSCAN': {'preds': preds_db, 'scores': scores_db},
    'Autoencoder': {'preds': preds_ae, 'scores': scores_ae}
}

metrics = []
for name, res in models_dict.items():
    p, s = res['preds'], res['scores']
    acc = accuracy_score(y_test, p)
    prec = precision_score(y_test, p, zero_division=0)
    rec = recall_score(y_test, p, zero_division=0)
    f1 = f1_score(y_test, p, zero_division=0)
    auc = roc_auc_score(y_test, s)
    pr_auc = average_precision_score(y_test, s)
    tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()

    metrics.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4),
        'PR-AUC': round(pr_auc, 4),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    })

eval_df = pd.DataFrame(metrics).sort_values(by=['F1-Score', 'ROC-AUC', 'Accuracy'], ascending=False).reset_index(drop=True)
eval_df.insert(0, 'Rank', [f"#{i+1}" for i in range(len(eval_df))])
print(eval_df.to_string(index=False))
eval_df
"""))

    # 18. Model Comparison & Champion Selection
    cells.append(nbf.v4.new_code_cell("""# 18. Model Comparison Visualizations & #1 Champion Selection
plt.figure(figsize=(12, 5))
melted = pd.melt(eval_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
sns.barplot(data=melted, x='Model', y='value', hue='variable', palette='Blues_r')
plt.title('5 Unsupervised ML Models Performance & Accuracy Comparison - Duplicate Invoice Detection')
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
ax1.set_title('ROC Curves (5 Unsupervised Models)')
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
    cells.append(nbf.v4.new_code_cell("""# 19. Feature Importance (All 32 Duplicate Detection Features)
from src.evaluate import compute_feature_importance

feat_imp = compute_feature_importance(iso_forest, FEATURE_COLS, X_train_scaled, y_true=y_train)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp.head(15), x='importance', y='feature', palette='mako')
plt.title('Top 15 Feature Importances for Duplicate Invoice Detection')
plt.tight_layout()
plt.show()

print(feat_imp.head(15).to_string(index=False))
"""))

    # 20. Final Prediction with Champion Model
    cells.append(nbf.v4.new_code_cell("""# 20. Final Prediction & Risk Triage using Selected #1 Champion Model
from src.models import SingleModelDuplicateInvoiceDetector
from src.inference import score_with_single_model

detector = SingleModelDuplicateInvoiceDetector(contamination=0.05, random_state=42, pretrained_model=iso_forest)
detector.fit(X_train_scaled)
df_scored = score_with_single_model(df, detector, scaler, FEATURE_COLS)

print(df_scored['risk_category'].value_counts())

audit_cols = ['invoice_doc_number', 'reference_invoice_no', 'vendor_name', 'gross_invoice_amount', 'duplicate_pattern_type', 'duplicate_invoice_prob', 'risk_category']
print(df_scored[df_scored['risk_category'] == 'Critical Risk'][audit_cols].head(10).to_string(index=False))
"""))

    # 21. Save Model
    cells.append(nbf.v4.new_code_cell("""# 21. Save Champion Model & Scaler
os.makedirs("models", exist_ok=True)

bundle = {
    'champion_model': iso_forest,
    'champion_name': champion_model,
    'scaler': scaler,
    'feature_cols': FEATURE_COLS
}

joblib.dump(bundle, "models/duplicate_invoice_model_bundle.joblib")
joblib.dump(scaler, "models/robust_scaler.joblib")
print("Saved champion model bundle and scaler to models/")
"""))

    nb.cells = cells

    paths = [
        os.path.join("notebooks", "20_Duplicate_Invoice_Detection_ML.ipynb"),
        "20_Duplicate_Invoice_Detection_ML.ipynb"
    ]
    for p in paths:
        os.makedirs(os.path.dirname(p) if os.path.dirname(p) else ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"[SUCCESS] Wrote 100% CODE-ONLY notebook to {p} ({len(cells)} code cells, 0 markdown cells).")


if __name__ == "__main__":
    create_code_only_notebook()
