import pandas as pd
import numpy as np
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data, CREDIT_RISK_FEATURE_COLS
from src.models import CreditRiskClusteringSuite
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, silhouette_score

df_raw = pd.read_csv('data/cache/credit_risk_raw_data_full.csv', low_memory=False)
df_clean = clean_and_engineer_features(df_raw)

data_dict = prepare_train_test_data(df_clean, test_size=0.20, random_state=42)
X_train_scaled = data_dict['X_train_scaled']
X_test_scaled = data_dict['X_test_scaled']
y_test = data_dict['y_test']

print(f"Test set total: {len(y_test):,}, Positive default rate: {y_test.mean()*100:.2f}%")

suite = CreditRiskClusteringSuite(n_clusters=4, contamination=0.20, random_state=42)
suite.train_all(X_train_scaled)

print("\n--- Evaluating with Standard Decision Boundary P(Default) >= 0.50 ---")
for name, spec, sub_cols in [
    ('K-Means Clustering', 'kmeans', 6),
    ('Gaussian Mixture (GMM)', 'gmm', 8),
    ('Mini-Batch K-Means', 'minibatch_kmeans', 6),
    ('Bisecting K-Means', 'bisecting_kmeans', 6),
    ('BIRCH Clustering', 'birch', 6)
]:
    meta = suite.cluster_meta[spec]
    model = suite.models[spec]
    
    if hasattr(model, 'predict'):
        raw_labels = model.predict(X_test_scaled[:, :sub_cols])
    else:
        raw_labels = model.fit_predict(X_test_scaled[:, :sub_cols])
        
    mapped_tiers = np.array([meta['rank_map'].get(int(c), 0) for c in raw_labels], dtype=int)
    scores = suite._compute_continuous_score(X_test_scaled, raw_labels, meta['rank_map'], meta['ordered_centroids'], sub_cols=sub_cols)
    
    # Standard Decision Boundary P(Default) >= 0.50
    preds_p50 = (scores >= 0.50).astype(int)
    # Cluster-based prediction: assigned to Highest Risk Tier (Tier 3)
    preds_tier3 = (mapped_tiers == 3).astype(int)
    
    acc_p50 = accuracy_score(y_test, preds_p50)
    prec_p50 = precision_score(y_test, preds_p50, zero_division=0)
    rec_p50 = recall_score(y_test, preds_p50, zero_division=0)
    f1_p50 = f1_score(y_test, preds_p50, zero_division=0)
    auc_p50 = roc_auc_score(y_test, scores)
    
    acc_t3 = accuracy_score(y_test, preds_tier3)
    prec_t3 = precision_score(y_test, preds_tier3, zero_division=0)
    rec_t3 = recall_score(y_test, preds_tier3, zero_division=0)
    f1_t3 = f1_score(y_test, preds_tier3, zero_division=0)
    
    print(f"\n{name}:")
    print(f"  [Standard P >= 0.50]: Acc: {acc_p50*100:.2f}% | Prec: {prec_p50*100:.2f}% | Rec: {rec_p50*100:.2f}% | F1: {f1_p50:.4f} | AUC: {auc_p50:.4f}")
    print(f"  [Cluster Tier 3 Only]: Acc: {acc_t3*100:.2f}% | Prec: {prec_t3*100:.2f}% | Rec: {rec_t3*100:.2f}% | F1: {f1_t3:.4f}")
