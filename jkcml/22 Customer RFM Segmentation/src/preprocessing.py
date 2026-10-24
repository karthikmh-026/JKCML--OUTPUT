"""
Preprocessing & Feature Engineering Module - Use Case 22: Customer RFM Segmentation (Clustering)
Transforms raw SAP Customer Master (KNA1, KNVV, KNB1), Billing (VBRK), Sales Orders (VBAK),
and Accounts Receivable (BSID) data into 32 domain-engineered numerical features for
Unsupervised Customer RFM Clustering (Group customers by behavior and value).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw SAP Customer Master, Billing, Sales Order, and Open AR data, and engineers
    32 domain-specific features across Recency (R), Frequency (F), Monetary Value (M),
    Purchase Consistency, Return/Cancellation Behavior, and Credit/AR Exposure.
    """
    df = df.copy()

    # 1. Clean string attributes across KNA1, KNVV, KNB1, VBRK
    str_cols = [
        'company_code', 'company_name', 'company_city', 'company_country', 'currency',
        'customer_id', 'customer_name', 'customer_city', 'postal_code', 'customer_region',
        'customer_country', 'account_group', 'account_group_desc', 'customer_classification',
        'industry_code', 'created_by', 'order_block', 'billing_block', 'delivery_block',
        'posting_block', 'deletion_flag', 'gstin_tax_no', 'sales_org', 'dist_channel',
        'division', 'customer_group', 'sales_district', 'price_list_type', 'price_group',
        'abc_class', 'sales_office', 'sales_group', 'payment_terms', 'recon_account',
        'payment_methods', 'channel_segment_desc', 'rural_urban_flag', 'metro_tier_flag',
        'distributor_type', 'dealer_tier'
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()
        else:
            df[col] = ''

    # 2. Clean numeric attributes across VBRK, VBAK, BSID
    num_cols = [
        'billing_doc_count', 'cancelled_or_return_bills', 'total_billed_net_value',
        'total_billed_tax_value', 'avg_bill_value', 'max_bill_value', 'std_bill_value',
        'distinct_billing_days', 'active_billing_months', 'sales_order_count',
        'total_order_net_value', 'avg_order_value', 'blocked_order_count',
        'open_ar_item_count', 'open_ar_balance', 'avg_ar_pending_days'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    # 3. Parse Dates & Determine Enterprise Snapshot Date
    last_bill_dt = pd.to_datetime(df['last_billing_date'], errors='coerce')
    first_bill_dt = pd.to_datetime(df['first_billing_date'], errors='coerce')
    last_order_dt = pd.to_datetime(df['last_order_date'], errors='coerce')
    first_order_dt = pd.to_datetime(df['first_order_date'], errors='coerce')
    last_ar_dt = pd.to_datetime(df['last_ar_posting_date'], errors='coerce')
    master_dt = pd.to_datetime(df['master_created_date'], errors='coerce')

    # Reference date is the maximum observed commercial transaction date in the SAP extract
    valid_max_dates = [
        dt.max() for dt in [last_bill_dt, last_order_dt, last_ar_dt]
        if dt.notna().any()
    ]
    snapshot_date = max(valid_max_dates) if valid_max_dates else pd.Timestamp("2025-06-30")

    # 4. Recency (R) Features
    df['recency_billing_days'] = (snapshot_date - last_bill_dt).dt.days.fillna(730.0).clip(0.0, 1095.0)
    df['recency_order_days'] = (snapshot_date - last_order_dt).dt.days.fillna(730.0).clip(0.0, 1095.0)
    ar_recency_days = (snapshot_date - last_ar_dt).dt.days.fillna(730.0).clip(0.0, 1095.0)

    df['recency_effective_days'] = np.minimum.reduce([
        df['recency_billing_days'].values,
        df['recency_order_days'].values,
        ar_recency_days.values
    ])

    earliest_activity_dt = first_bill_dt.combine_first(first_order_dt).combine_first(master_dt)
    df['customer_tenure_days'] = (snapshot_date - earliest_activity_dt).dt.days.fillna(365.0).clip(1.0, 7300.0)
    df['active_span_days'] = (last_bill_dt - first_bill_dt).dt.days.fillna(0.0).clip(0.0, 3650.0)
    df['recency_to_tenure_ratio'] = np.clip(
        df['recency_effective_days'] / np.maximum(1.0, df['customer_tenure_days']),
        0.0, 2.0
    )

    # 5. Frequency (F) & Purchase Cadence Features
    df['log_billing_frequency'] = np.log1p(np.maximum(0.0, df['billing_doc_count']))
    df['log_order_frequency'] = np.log1p(np.maximum(0.0, df['sales_order_count']))

    df['monthly_billing_velocity'] = np.where(
        df['active_billing_months'] > 0,
        df['billing_doc_count'] / df['active_billing_months'],
        0.0
    ).clip(0.0, 500.0)

    df['inter_purchase_interval_days'] = np.where(
        df['billing_doc_count'] > 1,
        df['active_span_days'] / (df['billing_doc_count'] - 1.0),
        df['recency_effective_days']
    ).clip(0.0, 1095.0)

    df['order_to_bill_conversion_ratio'] = np.where(
        df['sales_order_count'] > 0,
        df['billing_doc_count'] / df['sales_order_count'],
        np.where(df['billing_doc_count'] > 0, 1.0, 0.0)
    ).clip(0.0, 5.0)

    # 6. Monetary Value (M) & Spend Volatility Features
    # Effective commercial monetary value combines billed revenue and open/unbilled order value
    effective_monetary = np.where(
        df['total_billed_net_value'] > 0,
        df['total_billed_net_value'],
        np.maximum(0.0, df['total_order_net_value'])
    )
    df['total_billed_net_value'] = np.maximum(0.0, df['total_billed_net_value'])
    df['log_monetary_value'] = np.log1p(np.maximum(0.0, effective_monetary))
    df['avg_bill_value'] = np.maximum(0.0, df['avg_bill_value'])
    df['log_avg_bill_value'] = np.log1p(df['avg_bill_value'])
    df['max_bill_value'] = np.maximum(0.0, df['max_bill_value'])
    df['std_bill_value'] = np.maximum(0.0, df['std_bill_value'])

    df['bill_value_cv'] = np.where(
        df['avg_bill_value'] > 0,
        df['std_bill_value'] / df['avg_bill_value'],
        0.0
    ).clip(0.0, 10.0)

    # Sales Organization Peer Monetary Z-Score
    so_mean = df.groupby('sales_org')['log_monetary_value'].transform('mean')
    so_std = df.groupby('sales_org')['log_monetary_value'].transform('std').fillna(1.0).replace(0, 1.0)
    df['sales_org_monetary_zscore'] = ((df['log_monetary_value'] - so_mean) / so_std).clip(-5.0, 5.0).fillna(0.0)

    # 7. Behavioral Friction, Returns, Credit & AR Exposure Features
    df['cancellation_return_rate'] = np.where(
        df['billing_doc_count'] > 0,
        df['cancelled_or_return_bills'] / df['billing_doc_count'],
        0.0
    ).clip(0.0, 1.0)

    df['ar_to_monetary_exposure_ratio'] = np.where(
        effective_monetary > 0,
        np.maximum(0.0, df['open_ar_balance']) / effective_monetary,
        np.where(df['open_ar_balance'] > 0, 1.0, 0.0)
    ).clip(0.0, 5.0)

    df['is_account_blocked_or_flagged'] = (
        (df['order_block'] != '') |
        (df['billing_block'] != '') |
        (df['delivery_block'] != '') |
        (df['posting_block'] != '') |
        (df['deletion_flag'] != '') |
        (df['blocked_order_count'] > 0)
    ).astype(int)

    # 8. Normalized R, F, M Component Scores (0 to 100 scale) & Composite RFM Score
    # Recency Score: 100 for 0 days recency, decaying smoothly to 0 at >=365 days
    df['recency_score_norm'] = np.round(
        100.0 * np.exp(-df['recency_effective_days'] / 90.0), 2
    ).clip(0.0, 100.0)

    # Frequency Score: scaled on log frequency relative to 99th percentile
    freq_combined = np.maximum(df['log_billing_frequency'], df['log_order_frequency'])
    f_p99 = max(float(np.percentile(freq_combined[freq_combined > 0], 99)) if (freq_combined > 0).any() else 5.0, 1.0)
    df['frequency_score_norm'] = np.round(
        np.clip((freq_combined / f_p99) * 100.0, 0.0, 100.0), 2
    )

    # Monetary Score: scaled on log monetary value relative to 99th percentile
    m_p99 = max(float(np.percentile(df['log_monetary_value'][df['log_monetary_value'] > 0], 99)) if (df['log_monetary_value'] > 0).any() else 16.0, 1.0)
    df['monetary_score_norm'] = np.round(
        np.clip((df['log_monetary_value'] / m_p99) * 100.0, 0.0, 100.0), 2
    )

    # Composite RFM Value Score [0 - 100] and Customer Churn / Behavioral Risk Index [0 - 100]
    raw_rfm_value = (
        0.30 * df['recency_score_norm'] +
        0.35 * df['frequency_score_norm'] +
        0.35 * df['monetary_score_norm'] -
        8.0 * df['cancellation_return_rate'] -
        6.0 * df['is_account_blocked_or_flagged']
    )
    df['rfm_composite_score'] = np.round(np.clip(raw_rfm_value, 0.0, 100.0), 2)
    df['rfm_churn_risk_index'] = np.round(100.0 - df['rfm_composite_score'], 2)

    # 9. Discrete 1-5 Quintile Scores (R_Score, F_Score, M_Score) for Executive SAP Reporting
    df['r_quintile'] = np.clip(np.ceil(df['recency_score_norm'] / 20.0), 1, 5).astype(int)
    df['f_quintile'] = np.clip(np.ceil(df['frequency_score_norm'] / 20.0), 1, 5).astype(int)
    df['m_quintile'] = np.clip(np.ceil(df['monetary_score_norm'] / 20.0), 1, 5).astype(int)
    df['rfm_tier_code'] = (
        df['r_quintile'].astype(str) +
        df['f_quintile'].astype(str) +
        df['m_quintile'].astype(str)
    )

    # 10. Post-Hoc Benchmark RFM Segment & Exception Flag (Y)
    # Strictly excluded from unsupervised clustering training; used ONLY for post-hoc evaluation & ranking of the 5 models
    seg_conditions = [
        (df['rfm_composite_score'] >= 65.0),
        (df['rfm_composite_score'] >= 45.0),
        (df['rfm_composite_score'] >= 22.0),
    ]
    seg_choices = [
        'Champions / High-Value Loyal',
        'Core Growth / Regular Customers',
        'At-Risk / Cooling Customers',
    ]
    df['benchmark_rfm_segment'] = np.select(seg_conditions, seg_choices, default='Dormant / Churned / Low-Value')

    # Binary post-hoc benchmark exception flag: At-Risk or Churned/Dormant or Blocked/High-AR-Delinquency Customer
    df['is_rfm_exception'] = (
        (df['rfm_churn_risk_index'] >= 78.0) |
        ((df['rfm_churn_risk_index'] >= 60.0) & ((df['is_account_blocked_or_flagged'] == 1) | (df['cancellation_return_rate'] >= 0.20)))
    ).astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extracts 32 numerical domain features for Unsupervised Customer RFM Clustering.
    """
    feature_cols = [
        'rfm_composite_score',
        'rfm_churn_risk_index',
        'recency_score_norm',
        'frequency_score_norm',
        'monetary_score_norm',
        'recency_billing_days',
        'recency_order_days',
        'recency_effective_days',
        'customer_tenure_days',
        'active_span_days',
        'recency_to_tenure_ratio',
        'billing_doc_count',
        'log_billing_frequency',
        'sales_order_count',
        'log_order_frequency',
        'distinct_billing_days',
        'active_billing_months',
        'monthly_billing_velocity',
        'inter_purchase_interval_days',
        'order_to_bill_conversion_ratio',
        'total_billed_net_value',
        'log_monetary_value',
        'avg_bill_value',
        'log_avg_bill_value',
        'max_bill_value',
        'std_bill_value',
        'bill_value_cv',
        'total_billed_tax_value',
        'total_order_net_value',
        'sales_org_monetary_zscore',
        'cancellation_return_rate',
        'ar_to_monetary_exposure_ratio'
    ]

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['is_rfm_exception'].values
    return X, y, feature_cols


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Extracts 32 domain features, performs stratified train/test split for post-hoc evaluation,
    and applies RobustScaler (fit strictly on X_train without labels).
    """
    X, y, feature_cols = get_feature_matrix(df)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'idx_train': idx_train,
        'idx_test': idx_test,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
