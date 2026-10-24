"""
Preprocessing & Feature Engineering Module - Use Case 20: Duplicate Invoice Detection
Transforms raw SAP Invoice & Procurement tables (RBKP, RSEG, LFA1, LFB1, EKPO, MAKT, T001, T001W)
into 32 structured domain features for Unsupervised Duplicate Invoice & Payment Anomaly Detection.
Preserves original database values and formats missing values as blank cells for Excel reporting.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw SAP incoming invoice data and engineers 32 domain-specific features
    for detecting duplicate invoices and potential duplicate payments.
    Preserves exact missing values as blank rather than synthetic default placeholders.
    """
    df = df.copy()

    # 1. Clean string identifiers & reference numbers
    vendor_clean = df['vendor_id'].fillna('').astype(str).str.strip()
    ref_clean = df['reference_invoice_no'].fillna('').astype(str).str.strip()
    mat_clean = df['material_id'].fillna('').astype(str).str.strip()
    po_clean = df['po_number'].fillna('').astype(str).str.strip()
    po_item_clean = df['po_item'].fillna('').astype(str).str.strip()

    df['vendor_id'] = vendor_clean
    df['reference_invoice_no'] = ref_clean
    df['material_id'] = mat_clean
    df['po_number'] = po_clean
    df['po_item'] = po_item_clean

    # Clean display text fields: ensure genuine missing values are empty strings (blank)
    text_fields = [
        'company_code', 'company_name', 'company_city', 'company_country', 'company_currency',
        'vendor_name', 'vendor_city', 'vendor_region', 'vendor_country', 'double_inv_check_flag',
        'plant', 'plant_name', 'plant_city', 'material_desc', 'material_type', 'material_group',
        'doc_type', 'posted_by_user', 'transaction_code', 'order_unit', 'currency',
        'payment_terms', 'payment_block', 'payment_method', 'invoice_status', 'debit_credit_ind'
    ]
    for tf in text_fields:
        if tf in df.columns:
            df[tf] = df[tf].fillna('').astype(str).str.strip()
            df[tf] = df[tf].replace({'None': '', 'nan': '', 'NaN': '', '<NA>': '', 'NULL': ''})

    # Normalized alphanumeric reference (strips spaces, hyphens, slashes, leading zeros for fuzzy collision detection)
    norm_ref = (
        ref_clean
        .str.upper()
        .str.replace(r'[^A-Z0-9]', '', regex=True)
        .str.lstrip('0')
    )
    df['norm_reference_no'] = np.where(ref_clean == '', '', np.where(norm_ref == '', ref_clean.str.upper(), norm_ref))

    # 2. Financial & Quantity Series for Feature Calculations
    gross_amt = pd.to_numeric(df['gross_invoice_amount'], errors='coerce').fillna(0.0).clip(lower=0.01)
    item_amt = pd.to_numeric(df['invoice_item_amount'], errors='coerce').fillna(gross_amt).clip(lower=0.01)
    tax_amt = pd.to_numeric(df['tax_amount'], errors='coerce').fillna(0.0).clip(lower=0.0)
    inv_qty = pd.to_numeric(df['invoiced_qty'], errors='coerce').fillna(1.0).replace(0, 1.0)
    po_qty = pd.to_numeric(df['po_quantity'], errors='coerce').fillna(inv_qty).replace(0, 1.0)
    po_net = pd.to_numeric(df['po_net_value'], errors='coerce').fillna(item_amt).replace(0, 0.01)

    # Rounded amount keys for exact & near-exact collision detection
    gross_amt_round = gross_amt.round(0)
    item_amt_round = item_amt.round(0)

    # 3. Parse Dates & Temporal Lags
    inv_dt = pd.to_datetime(df['invoice_date'], errors='coerce')
    post_dt = pd.to_datetime(df['posting_date'], errors='coerce').fillna(inv_dt)
    entry_dt = pd.to_datetime(df['entry_date'], errors='coerce').fillna(post_dt)
    inv_dt = inv_dt.fillna(post_dt)

    inv_date_str = inv_dt.dt.strftime('%Y-%m-%d').fillna('')

    # 4. Multi-Key Duplicate Collision Frequencies
    # (a) Exact Vendor + Reference Invoice Number (XBLNR) - only count when reference is not blank
    has_ref = (ref_clean != '') & (ref_clean.str.len() >= 2)
    ref_counts = df[has_ref].groupby([vendor_clean[has_ref], ref_clean[has_ref]])['invoice_doc_number'].transform('count')
    df['exact_vendor_ref_count'] = 1
    df.loc[has_ref, 'exact_vendor_ref_count'] = ref_counts.fillna(1).astype(int)

    # (b) Normalized / Fuzzy Vendor + Alphanumeric Reference Number
    norm_has_ref = (df['norm_reference_no'] != '') & (df['norm_reference_no'].str.len() >= 2)
    norm_counts = df[norm_has_ref].groupby([vendor_clean[norm_has_ref], df.loc[norm_has_ref, 'norm_reference_no']])['invoice_doc_number'].transform('count')
    df['norm_vendor_ref_count'] = 1
    df.loc[norm_has_ref, 'norm_vendor_ref_count'] = norm_counts.fillna(1).astype(int)

    # (c) Exact Vendor + Gross Amount + Invoice Date (valid dates only)
    has_date = (inv_date_str != '')
    vad_counts = df[has_date].groupby([vendor_clean[has_date], gross_amt_round[has_date], inv_date_str[has_date]])['invoice_doc_number'].transform('count')
    df['vendor_amt_date_count'] = 1
    df.loc[has_date, 'vendor_amt_date_count'] = vad_counts.fillna(1).astype(int)

    # (d) Exact Vendor + Gross Amount across any date
    df['vendor_exact_amt_count'] = df.groupby([vendor_clean, gross_amt_round])['invoice_doc_number'].transform('count')

    # (e) Same PO + PO Item + Line Item Amount (only when PO item is not blank)
    has_po = (po_clean != '') & (po_item_clean != '')
    po_counts = df[has_po].groupby([po_clean[has_po], po_item_clean[has_po], item_amt_round[has_po]])['invoice_doc_number'].transform('count')
    df['po_item_exact_amt_count'] = 1
    df.loc[has_po, 'po_item_exact_amt_count'] = po_counts.fillna(1).astype(int)

    # (f) Same Vendor + Material + Gross Amount (only when material is not blank)
    has_mat = (mat_clean != '')
    mat_counts = df[has_mat].groupby([vendor_clean[has_mat], mat_clean[has_mat], gross_amt_round[has_mat]])['invoice_doc_number'].transform('count')
    df['vendor_mat_amt_count'] = 1
    df.loc[has_mat, 'vendor_mat_amt_count'] = mat_counts.fillna(1).astype(int)

    # 5. Binary Duplicate Collision Indicators
    df['is_exact_ref_dup'] = (df['exact_vendor_ref_count'] > 1).astype(int)
    df['is_fuzzy_ref_dup'] = ((df['norm_vendor_ref_count'] > 1) & (df['norm_reference_no'].str.len() >= 2)).astype(int)
    df['is_same_vendor_amt_date'] = (df['vendor_amt_date_count'] > 1).astype(int)
    df['is_same_po_item_amt'] = (df['po_item_exact_amt_count'] > 1).astype(int)

    # 6. Temporal Proximity to Nearest Identical Vendor+Amount Invoice & PO+Item Invoice
    inv_dt_valid = inv_dt.fillna(pd.Timestamp('2025-01-01'))
    inv_dt_num = (inv_dt_valid.astype('int64') // (10**9 * 86400)).astype(float)
    df['_inv_dt_num'] = inv_dt_num
    df['_gross_amt_round'] = gross_amt_round

    df_sorted_va = df.sort_values(['vendor_id', '_gross_amt_round', '_inv_dt_num'])
    prev_va_diff = df_sorted_va.groupby(['vendor_id', '_gross_amt_round'])['_inv_dt_num'].diff().abs()
    next_va_diff = df_sorted_va.groupby(['vendor_id', '_gross_amt_round'])['_inv_dt_num'].diff(-1).abs()
    min_va_days = np.minimum(prev_va_diff.fillna(9999), next_va_diff.fillna(9999))
    df.loc[df_sorted_va.index, '_min_va_days'] = min_va_days

    df['days_since_same_vendor_amt'] = np.where(
        df['vendor_exact_amt_count'] > 1,
        1.0 / (1.0 + df['_min_va_days'].clip(0, 365)),
        0.0
    )

    df_sorted_po = df.sort_values(['po_number', 'po_item', '_inv_dt_num'])
    prev_po_diff = df_sorted_po.groupby(['po_number', 'po_item'])['_inv_dt_num'].diff().abs()
    next_po_diff = df_sorted_po.groupby(['po_number', 'po_item'])['_inv_dt_num'].diff(-1).abs()
    min_po_days = np.minimum(prev_po_diff.fillna(9999), next_po_diff.fillna(9999))
    df.loc[df_sorted_po.index, '_min_po_days'] = min_po_days

    has_valid_po = (df['po_number'] != '') & (df['po_item'] != '')
    po_inv_cnt = pd.Series(1, index=df.index)
    if has_valid_po.any():
        po_inv_cnt.loc[has_valid_po] = df[has_valid_po].groupby(['po_number', 'po_item'])['invoice_doc_number'].transform('count')

    df['po_item_inv_frequency'] = po_inv_cnt
    df['days_since_same_po_item'] = np.where(
        (po_inv_cnt > 1) & has_valid_po,
        1.0 / (1.0 + df['_min_po_days'].clip(0, 365)),
        0.0
    )

    # 7. PO Over-Billing & Cumulative Invoicing Ratios
    df['inv_to_po_value_ratio'] = (item_amt / po_net).clip(0, 25.0)
    cum_po_item_amt = pd.Series(item_amt, index=df.index)
    if has_valid_po.any():
        cum_po_item_amt.loc[has_valid_po] = df[has_valid_po].groupby(['po_number', 'po_item'])['invoice_item_amount'].transform('sum')
    df['cum_po_invoiced_ratio'] = (cum_po_item_amt / po_net).clip(0, 50.0)
    df['is_po_over_invoiced'] = ((df['cum_po_invoiced_ratio'] > 1.05) & (po_inv_cnt > 1) & has_valid_po).astype(int)
    df['inv_to_po_qty_ratio'] = (inv_qty / po_qty).clip(0, 25.0)

    # 8. Financial Scale, Unit Price & Tax Ratios
    df['log_gross_amount'] = np.log1p(gross_amt)
    df['log_item_amount'] = np.log1p(item_amt)
    df['tax_to_gross_ratio'] = (tax_amt / gross_amt).clip(0, 1.0)
    df['effective_inv_unit_price'] = (item_amt / inv_qty).clip(0, 1e8)

    # 9. Statistical Distribution & Vendor/Material Z-Scores
    v_mean = df.groupby('vendor_id')['gross_invoice_amount'].transform('mean').fillna(gross_amt)
    v_std = df.groupby('vendor_id')['gross_invoice_amount'].transform('std').fillna(1.0).replace(0, 1.0)
    df['inv_po_price_zscore'] = ((gross_amt - v_mean) / v_std).clip(-10, 10).fillna(0.0)

    v_cnt = df.groupby('vendor_id')['invoice_doc_number'].transform('count')
    df['vendor_inv_frequency'] = np.log1p(v_cnt)

    # 10. Operational & Audit Control Features
    df['posting_lag_days'] = (post_dt - inv_dt).dt.days.fillna(0).clip(-30, 365)
    df['entry_lag_days'] = (entry_dt - post_dt).dt.days.fillna(0).clip(-30, 365)
    df['has_double_inv_check_flag'] = (df['double_inv_check_flag'].astype(str).str.upper() == 'X').astype(int)
    df['is_payment_blocked'] = (df['payment_block'].fillna('').astype(str).str.strip() != '').astype(int)
    df['is_weekend_or_monthend'] = ((post_dt.dt.dayofweek >= 5) | (post_dt.dt.day >= 25)).astype(int)

    user_cnt = df.groupby('posted_by_user')['invoice_doc_number'].transform('count')
    df['user_posting_intensity'] = np.log1p(user_cnt)

    # 11. Composite Duplicate Similarity Score (0 to 100 scale)
    raw_dup_score = (
        30.0 * df['is_exact_ref_dup'] +
        18.0 * df['is_fuzzy_ref_dup'] +
        22.0 * df['is_same_vendor_amt_date'] +
        18.0 * df['is_same_po_item_amt'] * df['is_po_over_invoiced'] +
        12.0 * df['days_since_same_vendor_amt'] +
        4.0 * np.minimum(5, df['exact_vendor_ref_count'] - 1) +
        4.0 * np.minimum(5, df['vendor_amt_date_count'] - 1)
    )
    df['dup_similarity_index'] = np.clip(raw_dup_score, 0.0, 100.0)

    # 12. Classify Human-Readable Duplicate Pattern Type for Executive Reporting
    pattern_conditions = [
        (df['is_exact_ref_dup'] == 1) & (df['is_same_vendor_amt_date'] == 1),
        (df['is_exact_ref_dup'] == 1),
        (df['is_fuzzy_ref_dup'] == 1) & (df['vendor_exact_amt_count'] > 1),
        (df['is_same_vendor_amt_date'] == 1) & (df['is_same_po_item_amt'] == 1) & (df['is_po_over_invoiced'] == 1),
        (df['is_same_vendor_amt_date'] == 1) & (df['vendor_amt_date_count'] >= 3),
        (df['is_same_po_item_amt'] == 1) & (df['is_po_over_invoiced'] == 1) & (df['days_since_same_vendor_amt'] >= 0.5)
    ]
    pattern_choices = [
        'Exact Vendor + Ref + Amount Duplicate',
        'Same Vendor + Reference No. Collision',
        'Fuzzy Reference + Identical Amount',
        'Same Day + PO Item Over-Billed Duplicate',
        'High-Frequency Same-Day Identical Amount',
        'Rapid Repeat PO Item Over-Billing'
    ]
    df['duplicate_pattern_type'] = np.select(pattern_conditions, pattern_choices, default='Normal Invoice')

    # 13. Post-Hoc Audit Benchmark Ground-Truth Exception Flag (Y)
    # Strictly excluded from unsupervised training; used ONLY to evaluate and rank the 5 ML models
    cond_exact_ref = (df['is_exact_ref_dup'] == 1)
    cond_fuzzy_amt = (df['is_fuzzy_ref_dup'] == 1) & (df['vendor_exact_amt_count'] > 1)
    cond_sameday_burst = (df['is_same_vendor_amt_date'] == 1) & (df['vendor_amt_date_count'] >= 3)
    cond_sameday_po_over = (
        (df['is_same_vendor_amt_date'] == 1) &
        (df['is_same_po_item_amt'] == 1) &
        (df['is_po_over_invoiced'] == 1)
    )

    df['is_duplicate_exception'] = (
        cond_exact_ref | cond_fuzzy_amt | cond_sameday_burst | cond_sameday_po_over
    ).astype(int)

    # Clean up temporary helper columns
    df.drop(columns=['_inv_dt_num', '_gross_amt_round', '_min_va_days', '_min_po_days'], inplace=True, errors='ignore')

    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extracts 32 numerical domain features for Unsupervised Duplicate Invoice Detection.
    """
    feature_cols = [
        'dup_similarity_index',
        'exact_vendor_ref_count',
        'norm_vendor_ref_count',
        'is_exact_ref_dup',
        'is_fuzzy_ref_dup',
        'vendor_amt_date_count',
        'is_same_vendor_amt_date',
        'vendor_exact_amt_count',
        'po_item_exact_amt_count',
        'is_same_po_item_amt',
        'vendor_mat_amt_count',
        'days_since_same_vendor_amt',
        'days_since_same_po_item',
        'cum_po_invoiced_ratio',
        'is_po_over_invoiced',
        'inv_to_po_value_ratio',
        'inv_to_po_qty_ratio',
        'gross_invoice_amount',
        'log_gross_amount',
        'invoice_item_amount',
        'log_item_amount',
        'tax_to_gross_ratio',
        'effective_inv_unit_price',
        'inv_po_price_zscore',
        'vendor_inv_frequency',
        'po_item_inv_frequency',
        'posting_lag_days',
        'entry_lag_days',
        'has_double_inv_check_flag',
        'is_payment_blocked',
        'is_weekend_or_monthend',
        'user_posting_intensity'
    ]

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['is_duplicate_exception'].values
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
