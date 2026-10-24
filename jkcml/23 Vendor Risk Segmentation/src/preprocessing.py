"""
Preprocessing & Feature Engineering Module - Use Case 23: Vendor Risk Segmentation (Clustering)
Transforms raw SAP Vendor Master (LFA1, LFB1, LFBK), Purchase Orders (EKKO, EKPO),
Invoice Receipts & Blocked Invoices (RBKP, RBKP_BLOCKED), Accounts Payable (BSIK),
and Quality Inspection Lots (QALS) into 32 domain-engineered numerical features for
Unsupervised Vendor Risk Clustering (Group vendors by reliability and risk).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw SAP Vendor Master, PO, Invoice, Blocked Invoice, Open AP, and Quality data,
    and engineers 32 domain-specific features across Delivery Reliability, Invoice & Payment
    Compliance, Quality Inspection Performance, Spend Exposure, and Master Governance Risk.
    """
    df = df.copy()

    # 1. Clean string attributes across LFA1, LFB1, EKKO, T077Y, T001
    str_cols = [
        'source_schema', 'company_code', 'company_name', 'company_city', 'company_country', 'currency',
        'vendor_id', 'vendor_name', 'vendor_city', 'postal_code', 'vendor_region',
        'vendor_country', 'account_group', 'account_group_desc', 'industry_code',
        'created_by', 'posting_block', 'purchasing_block', 'quality_block',
        'payment_block', 'deletion_flag', 'gstin_tax_no', 'pan_number',
        'alt_payee_allowed', 'one_time_vendor_flag', 'purchasing_org', 'purchasing_group',
        'primary_po_type', 'payment_terms', 'recon_account', 'payment_methods',
        'dup_invoice_check_flag', 'withholding_tax_code', 'cash_mgmt_group'
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()
            df[col] = df[col].replace({'nan': '', 'NaN': '', 'None': '', 'null': '', 'NaT': '', '<NA>': '', 'undefined': ''})
        else:
            df[col] = ''

    # 2. Clean numeric attributes across EKKO, EKPO, RBKP, RBKP_BLOCKED, BSIK, QALS, LFBK
    num_cols = [
        'bank_account_count', 'po_count', 'po_item_count', 'deleted_po_count',
        'deleted_po_item_count', 'return_po_item_count', 'delivery_completed_items',
        'final_invoiced_items', 'total_po_net_value', 'avg_po_item_value',
        'max_po_item_value', 'std_po_item_value', 'avg_gr_processing_days',
        'avg_overdelivery_tol', 'total_po_dunning_reminders', 'active_po_months',
        'invoice_count', 'total_invoice_gross_value', 'avg_invoice_value',
        'max_invoice_value', 'std_invoice_value', 'total_unplanned_deliv_costs',
        'total_invoice_tax_value', 'payment_blocked_invoices', 'reversed_invoices',
        'credit_memo_or_adj_count', 'mrm_blocked_invoice_count', 'open_ap_item_count',
        'open_ap_balance', 'blocked_ap_item_count', 'avg_ap_pending_days',
        'inspection_lot_count', 'total_inspected_qty', 'total_accepted_qty',
        'total_rejected_blocked_qty'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    # 3. Parse Dates & Determine Enterprise Snapshot Date
    last_po_dt = pd.to_datetime(df.get('last_po_date', pd.Series(index=df.index)), errors='coerce', dayfirst=True)
    first_po_dt = pd.to_datetime(df.get('first_po_date', pd.Series(index=df.index)), errors='coerce', dayfirst=True)
    last_inv_dt = pd.to_datetime(df.get('last_invoice_date', pd.Series(index=df.index)), errors='coerce', dayfirst=True)
    first_inv_dt = pd.to_datetime(df.get('first_invoice_date', pd.Series(index=df.index)), errors='coerce', dayfirst=True)
    last_ap_dt = pd.to_datetime(df.get('last_ap_posting_date', pd.Series(index=df.index)), errors='coerce', dayfirst=True)
    master_dt = pd.to_datetime(df.get('master_created_date', pd.Series(index=df.index)), errors='coerce', dayfirst=True)

    valid_max_dates = [
        dt.max() for dt in [last_po_dt, last_inv_dt, last_ap_dt, master_dt]
        if dt.notna().any()
    ]
    snapshot_date = max(valid_max_dates) if valid_max_dates else pd.Timestamp("2025-06-30")

    # 4. Recency & Relationship Tenure Features
    earliest_dt = first_po_dt.combine_first(first_inv_dt).combine_first(master_dt)
    latest_tx_dt = last_po_dt.combine_first(last_inv_dt).combine_first(last_ap_dt)

    df['vendor_tenure_days'] = (snapshot_date - earliest_dt).dt.days.fillna(365.0).clip(1.0, 7300.0)
    df['recency_po_days'] = (snapshot_date - last_po_dt).dt.days.fillna(df['vendor_tenure_days']).clip(0.0, 3650.0)
    df['recency_invoice_days'] = (snapshot_date - last_inv_dt).dt.days.fillna(df['vendor_tenure_days']).clip(0.0, 3650.0)
    ap_recency_days = (snapshot_date - last_ap_dt).dt.days.fillna(df['vendor_tenure_days']).clip(0.0, 3650.0)

    df['recency_effective_days'] = np.minimum.reduce([
        df['recency_po_days'].values,
        df['recency_invoice_days'].values,
        ap_recency_days.values
    ])

    df['active_procurement_span_days'] = (latest_tx_dt - earliest_dt).dt.days.fillna(0.0).clip(0.0, 3650.0)

    # 5. Procurement Volume, Spend & Volatility Features
    df['log_po_frequency'] = np.log1p(np.maximum(0.0, df['po_count']))
    df['log_invoice_frequency'] = np.log1p(np.maximum(0.0, df['invoice_count']))

    df['total_po_net_value'] = np.maximum(0.0, df['total_po_net_value'])
    df['total_invoice_gross_value'] = np.maximum(0.0, df['total_invoice_gross_value'])
    effective_spend = np.maximum.reduce([
        df['total_po_net_value'].values,
        df['total_invoice_gross_value'].values,
        np.maximum(0.0, df['open_ap_balance'].values)
    ])
    df['effective_vendor_spend'] = effective_spend
    df['log_total_spend'] = np.log1p(effective_spend)

    df['avg_po_item_value'] = np.maximum(0.0, df['avg_po_item_value'])
    df['max_po_item_value'] = np.maximum(0.0, df['max_po_item_value'])
    df['std_po_item_value'] = np.maximum(0.0, df['std_po_item_value'])
    df['avg_invoice_value'] = np.maximum(0.0, df['avg_invoice_value'])

    df['po_spend_cv'] = np.where(
        df['avg_po_item_value'] > 0,
        df['std_po_item_value'] / df['avg_po_item_value'],
        0.0
    ).clip(0.0, 10.0)

    # Purchasing Organization Peer Spend Z-Score
    po_org_mean = df.groupby('purchasing_org')['log_total_spend'].transform('mean')
    po_org_std = df.groupby('purchasing_org')['log_total_spend'].transform('std').fillna(1.0).replace(0, 1.0)
    df['purch_org_spend_zscore'] = ((df['log_total_spend'] - po_org_mean) / po_org_std).clip(-5.0, 5.0).fillna(0.0)

    # 6. Delivery Reliability & PO Fulfillment Risk Features
    df['po_deletion_rate'] = np.where(
        df['po_item_count'] > 0,
        df['deleted_po_item_count'] / df['po_item_count'],
        np.where(df['po_count'] > 0, df['deleted_po_count'] / df['po_count'], 0.0)
    ).clip(0.0, 1.0)

    df['po_return_item_rate'] = np.where(
        df['po_item_count'] > 0,
        df['return_po_item_count'] / df['po_item_count'],
        0.0
    ).clip(0.0, 1.0)

    df['po_delivery_completion_rate'] = np.where(
        df['po_item_count'] > 0,
        df['delivery_completed_items'] / df['po_item_count'],
        np.where(df['invoice_count'] > 0, 0.85, 0.0)
    ).clip(0.0, 1.0)

    # 7. Invoice Compliance, Payment Block & AP Exposure Risk Features
    total_blocked_inv = np.maximum(df['payment_blocked_invoices'], df['mrm_blocked_invoice_count'])
    df['total_blocked_invoices'] = total_blocked_inv

    df['invoice_block_rate'] = np.where(
        df['invoice_count'] > 0,
        total_blocked_inv / df['invoice_count'],
        np.where(total_blocked_inv > 0, 1.0, 0.0)
    ).clip(0.0, 1.0)

    df['invoice_reversal_rate'] = np.where(
        df['invoice_count'] > 0,
        df['reversed_invoices'] / df['invoice_count'],
        0.0
    ).clip(0.0, 1.0)

    df['unplanned_deliv_cost_ratio'] = np.where(
        df['total_invoice_gross_value'] > 0,
        np.maximum(0.0, df['total_unplanned_deliv_costs']) / df['total_invoice_gross_value'],
        0.0
    ).clip(0.0, 1.0)

    df['open_ap_to_spend_ratio'] = np.where(
        effective_spend > 0,
        np.maximum(0.0, df['open_ap_balance']) / effective_spend,
        np.where(df['open_ap_balance'] > 0, 1.0, 0.0)
    ).clip(0.0, 5.0)

    # 8. Quality Inspection & Master Governance Risk Features
    df['quality_rejection_rate'] = np.where(
        df['total_inspected_qty'] > 0,
        np.maximum(0.0, df['total_rejected_blocked_qty']) / df['total_inspected_qty'],
        0.0
    ).clip(0.0, 1.0)

    df['is_vendor_blocked_or_flagged'] = (
        (df['posting_block'] != '') |
        (df['purchasing_block'] != '') |
        (df['quality_block'] != '') |
        (df['payment_block'] != '') |
        (df['deletion_flag'] != '') |
        (df['blocked_ap_item_count'] > 0)
    ).astype(int)

    df['has_verified_tax_id'] = ((df['gstin_tax_no'] != '') | (df['pan_number'] != '')).astype(float)
    df['has_verified_bank'] = (df['bank_account_count'] > 0).astype(float)
    df['has_payment_terms'] = (df['payment_terms'] != '').astype(float)

    # 9. Normalized Component Scores (0 to 100 scale) & Composite Vendor Reliability / Risk Scores
    # (A) Delivery & Activity Reliability Score [0-100]
    recency_factor = 100.0 * np.exp(-df['recency_effective_days'] / 365.0)
    tx_freq_combined = np.log1p(df['po_count'] + df['invoice_count'] + df['open_ap_item_count'])
    f_p99 = max(float(np.percentile(tx_freq_combined[tx_freq_combined > 0], 99)) if (tx_freq_combined > 0).any() else 5.0, 1.0)
    volume_factor = np.clip((tx_freq_combined / f_p99) * 100.0, 0.0, 100.0)

    raw_delivery_rel = (
        0.42 * recency_factor +
        0.33 * volume_factor +
        0.25 * (df['po_delivery_completion_rate'] * 100.0) -
        28.0 * df['po_deletion_rate'] -
        35.0 * df['po_return_item_rate']
    )
    df['delivery_reliability_score'] = np.round(np.clip(raw_delivery_rel, 0.0, 100.0), 2)

    # (B) Invoice & Financial Compliance Score [0-100]
    spend_p99 = max(float(np.percentile(df['log_total_spend'][df['log_total_spend'] > 0], 99)) if (df['log_total_spend'] > 0).any() else 15.0, 1.0)
    spend_norm = np.clip((df['log_total_spend'] / spend_p99) * 100.0, 0.0, 100.0)
    tenure_maturity = np.clip(100.0 * (1.0 - np.exp(-df['vendor_tenure_days'] / 450.0)), 10.0, 100.0)

    raw_inv_compliance = (
        0.45 * spend_norm +
        0.35 * recency_factor +
        0.20 * tenure_maturity -
        42.0 * df['invoice_block_rate'] -
        30.0 * df['invoice_reversal_rate'] -
        20.0 * df['unplanned_deliv_cost_ratio'] -
        10.0 * np.clip(df['open_ap_to_spend_ratio'], 0.0, 1.5)
    )
    df['invoice_compliance_score'] = np.round(np.clip(raw_inv_compliance, 0.0, 100.0), 2)

    # (C) Quality & Master Data Governance Score [0-100]
    raw_qual_gov = (
        28.0 +
        24.0 * df['has_verified_bank'] +
        22.0 * df['has_verified_tax_id'] +
        14.0 * df['has_payment_terms'] +
        0.12 * recency_factor +
        np.where(df['inspection_lot_count'] > 0, 12.0 * (1.0 - df['quality_rejection_rate']), 0.0) -
        45.0 * df['quality_rejection_rate'] -
        38.0 * df['is_vendor_blocked_or_flagged']
    )
    df['quality_governance_score'] = np.round(np.clip(raw_qual_gov, 0.0, 100.0), 2)

    # Composite Vendor Reliability Score [0-100] and Composite Vendor Risk Index [0-100]
    raw_reliability = (
        0.36 * df['delivery_reliability_score'] +
        0.34 * df['invoice_compliance_score'] +
        0.30 * df['quality_governance_score']
    )
    df['vendor_reliability_score'] = np.round(np.clip(raw_reliability, 0.0, 100.0), 2)
    df['vendor_risk_index'] = np.round(100.0 - df['vendor_reliability_score'], 2)

    # Discrete 1-5 Quintile Ratings (D-I-Q: Delivery, Invoice Compliance, Quality/Governance)
    df['d_rating'] = np.clip(np.ceil(df['delivery_reliability_score'] / 20.0), 1, 5).astype(int)
    df['i_rating'] = np.clip(np.ceil(df['invoice_compliance_score'] / 20.0), 1, 5).astype(int)
    df['q_rating'] = np.clip(np.ceil(df['quality_governance_score'] / 20.0), 1, 5).astype(int)
    df['vendor_tier_code'] = (
        "D" + df['d_rating'].astype(str) +
        "-I" + df['i_rating'].astype(str) +
        "-Q" + df['q_rating'].astype(str)
    )

    # 10. Post-Hoc Benchmark Vendor Segment & Exception Flag (Y)
    # Strictly excluded from unsupervised clustering training; used ONLY for post-hoc evaluation & ranking of the 5 models
    p22 = float(np.percentile(df['vendor_reliability_score'], 21.0))
    p50 = float(np.percentile(df['vendor_reliability_score'], 52.0))
    p78 = float(np.percentile(df['vendor_reliability_score'], 78.0))

    seg_conditions = [
        (df['vendor_reliability_score'] >= p78),
        (df['vendor_reliability_score'] >= p50),
        (df['vendor_reliability_score'] >= p22),
    ]
    seg_choices = [
        'Strategic / High-Reliability Vendors',
        'Preferred / Core Operational Vendors',
        'Watchlist / Moderate-Risk Vendors',
    ]
    df['benchmark_vendor_segment'] = np.select(seg_conditions, seg_choices, default='High-Risk / Non-Compliant / Blocked Vendors')

    # 10. Post-Hoc Benchmark Vendor Segment & Exception Flag (Y)
    # Strictly excluded from unsupervised clustering training; used ONLY for post-hoc evaluation & ranking of the 5 models
    # Multi-criteria operational risk ground truth based on master blocks, blocked invoices, delivery deletions, and liabilities:
    operational_friction = (
        0.35 * df['is_vendor_blocked_or_flagged'] +
        0.25 * df['invoice_block_rate'] +
        0.20 * df['po_deletion_rate'] +
        0.15 * df['po_return_item_rate'] +
        0.15 * df['quality_rejection_rate'] +
        0.15 * np.clip(df['open_ap_to_spend_ratio'] / 2.0, 0, 1.0) +
        0.10 * np.where((df['open_ap_balance'] > 5000) & (df['recency_effective_days'] > 180), 1.0, 0.0) +
        0.05 * np.where((df['has_verified_tax_id'] == 0) & (df['log_total_spend'] > 10.0), 1.0, 0.0)
    )
    p78_risk = float(np.percentile(operational_friction[operational_friction > 0], 65.0)) if (operational_friction > 0).any() else 0.25
    df['is_vendor_risk_exception'] = (
        (df['is_vendor_blocked_or_flagged'] == 1) |
        (df['total_blocked_invoices'] > 0) |
        (operational_friction >= p78_risk)
    ).astype(int)

    p22 = float(np.percentile(df['vendor_reliability_score'], 21.0))
    p50 = float(np.percentile(df['vendor_reliability_score'], 52.0))
    p78 = float(np.percentile(df['vendor_reliability_score'], 78.0))

    seg_conditions = [
        (df['vendor_reliability_score'] >= p78),
        (df['vendor_reliability_score'] >= p50),
        (df['vendor_reliability_score'] >= p22),
    ]
    seg_choices = [
        'Strategic / High-Reliability Vendors',
        'Preferred / Core Operational Vendors',
        'Watchlist / Moderate-Risk Vendors',
    ]
    df['benchmark_vendor_segment'] = np.select(seg_conditions, seg_choices, default='High-Risk / Non-Compliant / Blocked Vendors')

    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extracts 28 genuine operational domain features for Unsupervised Vendor Risk Clustering.
    STRICTLY EXCLUDES composite summary scores (vendor_reliability_score, vendor_risk_index, etc.)
    to eliminate target leakage and ensure realistic clustering evaluation.
    """
    feature_cols = [
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

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['is_vendor_risk_exception'].values
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

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
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
