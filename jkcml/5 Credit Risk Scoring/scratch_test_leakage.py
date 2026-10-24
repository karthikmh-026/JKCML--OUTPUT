import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, silhouette_score

df = pd.read_csv('data/cache/credit_risk_raw_data_full.csv', low_memory=False)
from src.preprocessing import clean_and_engineer_features, CREDIT_RISK_FEATURE_COLS

df_clean = clean_and_engineer_features(df)

# Let's inspect the correlation of credit_risk_index with is_credit_risk_exception
print("Correlation with benchmark exception:")
corrs = df_clean[CREDIT_RISK_FEATURE_COLS].apply(lambda c: c.corr(df_clean['is_credit_risk_exception'])).sort_values(ascending=False)
print(corrs.head(10))

# Let's exclude synthetic formula columns from the clustering features to avoid circular leakage
independent_features = [
    "overdue_ratio", "max_overdue_days", "avg_overdue_days", "severe_overdue_ratio",
    "open_ar_log_val", "overdue_ar_log_val", "blocked_ar_rate", "blocked_balance_ratio",
    "avg_clearing_delay_days", "max_clearing_delay_days", "late_settlement_rate",
    "credit_limit_utilization", "is_credit_limit_breached", "unsecured_credit_gap",
    "billing_volatility_ratio", "order_to_billing_ratio", "blocked_order_rate", "return_memo_rate",
    "has_central_posting_block", "has_cocode_posting_block", "payment_terms_risk_score",
    "settlement_velocity_index", "cash_flow_strain_ratio"
]
print(f"\nUsing {len(independent_features)} independent domain features.")

X = df_clean[independent_features].values
y = df_clean['is_credit_risk_exception'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test 5 unsupervised clustering models on independent features
models = {
    'K-Means Clustering': KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42),
    'Mini-Batch K-Means': MiniBatchKMeans(n_clusters=4, batch_size=2048, n_init=5, random_state=42),
    'Bisecting K-Means': BisectingKMeans(n_clusters=4, init='k-means++', random_state=42),
    'Gaussian Mixture (GMM)': GaussianMixture(n_components=4, covariance_type='diag', max_iter=150, random_state=42),
    'BIRCH Clustering': Birch(n_clusters=4, threshold=0.5, branching_factor=50)
}

print("\n--- Model Evaluation on Independent Multi-Dimensional Financial Features ---")
for name, m in models.items():
    if hasattr(m, 'fit_predict'):
        train_labels = m.fit_predict(X_train_scaled)
    else:
        train_labels = m.fit(X_train_scaled).predict(X_train_scaled)
    
    # Calculate centroids
    centroids = np.zeros((4, X_train_scaled.shape[1]))
    for c in range(4):
        mask = (train_labels == c)
        if np.any(mask):
            centroids[c] = X_train_scaled[mask].mean(axis=0)
    
    # Risk signal: overdue_ratio (idx 0) + severe_overdue_ratio (idx 3) + overdue_ar_log_val (idx 5)
    risk_signal = centroids[:, 0] + centroids[:, 3] + centroids[:, 5]
    sorted_ids = np.argsort(risk_signal)
    rank_map = {int(orig): int(new) for new, orig in enumerate(sorted_ids)}
    
    test_labels = m.predict(X_test_scaled)
    mapped_test = np.array([rank_map[int(c)] for c in test_labels])
    
    # Distance-based score
    c_prime = centroids[sorted_ids[0]]
    c_risk = centroids[sorted_ids[-1]]
    d_p = np.linalg.norm(X_test_scaled - c_prime, axis=1)
    d_r = np.linalg.norm(X_test_scaled - c_risk, axis=1)
    rel_affinity = d_p / np.maximum(1e-6, d_p + d_r)
    scores = 0.5 * (mapped_test / 3.0) + 0.5 * rel_affinity
    
    # Natural cluster-based prediction: assigned to highest risk cluster (Tier 3) or high affinity
    preds = (mapped_test >= 2).astype(int)  # Tier 2 (Watchlist) & Tier 3 (High Risk)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, scores)
    
    print(f"{name:25s} | Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}")
