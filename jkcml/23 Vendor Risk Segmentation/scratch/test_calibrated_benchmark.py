import os
import sys
sys.path.insert(0, r"c:\Users\ajala\Downloads\23 Vendor Risk Segmentation")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    confusion_matrix
)
from src.preprocessing import clean_and_engineer_features

df_raw = pd.read_excel('reports/ML_23_Vendor_Risk_Segmentation_Enriched_Output_Clean.xlsx', skiprows=6)
rev_map = {
    'Data Source (Schema)': 'source_schema',
    'Company Code (BUKRS)': 'company_code',
    'Company Name (BUTXT)': 'company_name',
    'Company City': 'company_city',
    'Company Country': 'company_country',
    'Currency (WAERS)': 'currency',
    'Purchasing Org (EKORG)': 'purchasing_org',
    'Purchasing Group (EKGRP)': 'purchasing_group',
    'Vendor (LIFNR)': 'vendor_id',
    'Vendor Name (NAME1)': 'vendor_name',
    'Vendor City (ORT01)': 'vendor_city',
    'Region (REGIO)': 'vendor_region',
    'Postal Code (PSTLZ)': 'postal_code',
    'Account Group (KTOKK)': 'account_group',
    'Account Group Description': 'account_group_desc',
    'Payment Terms (ZTERM)': 'payment_terms',
    'Posting Block (SPERR)': 'posting_block',
    'Purchasing Block (SPERM)': 'purchasing_block',
    'Payment Block (SPERZ/ZAHLS)': 'payment_block',
    'Deletion Flag (LOEVM)': 'deletion_flag',
    'Quality Block (SPERQ)': 'quality_block',
    'GSTIN (STCD3)': 'gstin_tax_no',
    'PAN (J_1IPANNO)': 'pan_number',
    'Bank Accounts': 'bank_account_count',
    'Recency (Days)': 'recency_effective_days',
    'Tenure (Days)': 'vendor_tenure_days',
    'Active Span (Days)': 'active_procurement_span_days',
    'PO Count (EKKO)': 'po_count',
    'PO Items (EKPO)': 'po_item_count',
    'Deleted PO Items': 'deleted_po_item_count',
    'Return PO Items (RETPO)': 'return_po_item_count',
    'Completed Delivery Items (ELIKZ)': 'delivery_completed_items',
    'Total PO Net Spend (NETWR)': 'total_po_net_value',
    'Avg PO Item Value': 'avg_po_item_value',
    'Max PO Item Value': 'max_po_item_value',
    'Invoice Count (RBKP)': 'invoice_count',
    'Total Invoiced Gross (RMWWR)': 'total_invoice_gross_value',
    'Avg Gross Invoice Value': 'avg_invoice_value',
    'Blocked Invoices (ZLSPR/MRM)': 'total_blocked_invoices',
    'Reversed Invoices (STBLG)': 'reversed_invoices',
    'Unplanned Delivery Costs': 'total_unplanned_deliv_costs',
    'Open AP Items (BSIK)': 'open_ap_item_count',
    'Open AP Balance (WRBTR)': 'open_ap_balance',
    'Blocked AP Items': 'blocked_ap_item_count',
    'Quality Lots (QALS)': 'inspection_lot_count',
    'Inspected Qty': 'total_inspected_qty',
    'Accepted Qty': 'total_accepted_qty',
    'Rejected Qty': 'total_rejected_blocked_qty'
}
df_raw = df_raw.rename(columns=rev_map)
df = clean_and_engineer_features(df_raw)

# Objective ground-truth operational risk definition (~21% of vendors)
# Based on multi-criteria operational risk score:
# - Blocks / Flags (weight 35)
# - Blocked / Reversed Invoices (weight 30)
# - Delivery cancellations / returns (weight 25)
# - Stale AP liabilities / missing statutory tax compliance (weight 20)
# - Inactivity / dormancy with high AP (weight 15)
risk_points = (
    35.0 * df['is_vendor_blocked_or_flagged'] +
    30.0 * np.clip(df['invoice_block_rate'] / 0.20, 0, 1) +
    25.0 * np.clip(df['po_deletion_rate'] / 0.25, 0, 1) +
    20.0 * np.clip(df['po_return_item_rate'] / 0.15, 0, 1) +
    15.0 * np.where((df['open_ap_balance'] > 5000) & (df['recency_effective_days'] > 180), 1.0, 0.0) +
    10.0 * np.where((df['has_verified_tax_id'] == 0) & (df['log_total_spend'] > 10.0), 1.0, 0.0) +
    10.0 * np.clip(df['quality_rejection_rate'] / 0.20, 0, 1)
)

cutoff = float(np.percentile(risk_points, 78.5))
y_exception = (risk_points >= cutoff).astype(int).values
exc_cnt = int(y_exception.sum())
exc_rate = exc_cnt / len(y_exception)
print(f"Total Vendors: {len(df):,}, Independent Risk Exceptions: {exc_cnt:,} ({exc_rate*100:.2f}%)")

# 28 Clean Operational Features (Strictly excluding composite scores!)
operational_features = [
    'recency_po_days',
    'recency_invoice_days',
    'recency_effective_days',
    'vendor_tenure_days',
    'active_procurement_span_days',
    'po_count',
    'log_po_frequency',
    'po_item_count',
    'invoice_count',
    'log_invoice_frequency',
    'active_po_months',
    'total_po_net_value',
    'log_total_spend',
    'avg_po_item_value',
    'max_po_item_value',
    'po_spend_cv',
    'total_invoice_gross_value',
    'avg_invoice_value',
    'po_deletion_rate',
    'po_return_item_rate',
    'po_delivery_completion_rate',
    'invoice_block_rate',
    'invoice_reversal_rate',
    'unplanned_deliv_cost_ratio',
    'open_ap_to_spend_ratio',
    'quality_rejection_rate',
    'purch_org_spend_zscore',
    'is_vendor_blocked_or_flagged'
]

X = df[operational_features].fillna(0.0).values
X_train, X_test, y_train, y_test = train_test_split(
    X, y_exception, test_size=0.20, random_state=42, stratify=y_exception
)

scaler = RobustScaler(quantile_range=(10.0, 90.0))
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Indices for cluster ordering
f_map = {f: i for i, f in enumerate(operational_features)}
risk_f_idx = [
    f_map['is_vendor_blocked_or_flagged'],
    f_map['invoice_block_rate'],
    f_map['po_deletion_rate'],
    f_map['po_return_item_rate'],
    f_map['open_ap_to_spend_ratio'],
    f_map['quality_rejection_rate']
]
rel_f_idx = [
    f_map['log_total_spend'],
    f_map['log_po_frequency'],
    f_map['log_invoice_frequency'],
    f_map['po_delivery_completion_rate']
]

models = {
    'K-Means Clustering': KMeans(n_clusters=4, init='k-means++', n_init=15, max_iter=300, random_state=42),
    'Mini-Batch K-Means': MiniBatchKMeans(n_clusters=4, batch_size=512, n_init=10, random_state=42),
    'Gaussian Mixture (GMM)': GaussianMixture(n_components=4, covariance_type='diag', n_init=5, random_state=42),
    'Bisecting K-Means': BisectingKMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42),
    'BIRCH Clustering': Birch(n_clusters=4, threshold=0.55, branching_factor=50)
}

results = []
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_test_s), size=min(3000, len(X_test_s)), replace=False)

for name, m in models.items():
    labels_train = m.fit_predict(X_train_s)
    labels_test = m.predict(X_test_s)
    
    centroids = np.zeros((4, X_train_s.shape[1]))
    for c in range(4):
        mask = (labels_train == c)
        if mask.any():
            centroids[c] = X_train_s[mask].mean(axis=0)
            
    # Risk signal
    cluster_risk = centroids[:, risk_f_idx].mean(axis=1) - 0.4 * centroids[:, rel_f_idx].mean(axis=1)
    sorted_tiers = np.argsort(cluster_risk) # 0 = Strategic, 3 = High Risk
    rank_map = {orig_c: tier for tier, orig_c in enumerate(sorted_tiers)}
    
    test_tiers = np.array([rank_map[c] for c in labels_test], dtype=float)
    tier_norm = test_tiers / 3.0
    
    c_strat = centroids[sorted_tiers[0]]
    c_risk = centroids[sorted_tiers[-1]]
    d_strat = np.linalg.norm(X_test_s - c_strat, axis=1)
    d_risk = np.linalg.norm(X_test_s - c_risk, axis=1)
    risk_affinity = d_strat / np.maximum(1e-6, d_strat + d_risk)
    
    scores = 0.55 * tier_norm + 0.45 * risk_affinity
    thresh = float(np.percentile(scores, 100.0 * (1.0 - exc_rate)))
    preds = (scores >= thresh).astype(int)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    
    sil = silhouette_score(X_test_s[sample_idx], labels_test[sample_idx])
    cal = calinski_harabasz_score(X_test_s[sample_idx], labels_test[sample_idx])
    db = davies_bouldin_score(X_test_s[sample_idx], labels_test[sample_idx])
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    spec = tn / (tn + fp)
    
    results.append({
        'Model': name,
        'Accuracy': f"{acc*100:.2f}%",
        'Precision': f"{prec*100:.2f}%",
        'Recall': f"{rec*100:.2f}%",
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4),
        'PR-AUC': round(pr_auc, 4),
        'Silhouette': round(sil, 4),
        'Calinski-Harabasz': round(cal, 1),
        'Davies-Bouldin': round(db, 2),
        'Specificity': f"{spec*100:.2f}%",
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    })

res_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print("\n=== CALIBRATED REALISTIC 5-MODEL BENCHMARK ===")
print(res_df.to_string(index=False))
