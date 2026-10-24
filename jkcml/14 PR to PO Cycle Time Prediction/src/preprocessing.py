"""
Preprocessing and 32-Feature Engineering Module - Use Case 14: PR to PO Cycle Time Prediction
Category: Regression
Objective: Estimate procurement processing time.

Handles:
1. Data cleaning and type parsing for SAP Purchase Requisitions across HANA_P2 and HANA_P3.
2. Strict missing value handling: if data is present, preserves it; if not present, marks as blank / clean defaults.
3. Domain-specific 32-feature engineering for Procurement Cycle Time Regression.
4. Computation of ground-truth target: actual_cycle_time_days = BEDAT - BADAT.
5. Robust scaling with RobustScaler and train/test split.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


FEATURE_COLUMNS = [
    'pr_item_value',
    'log_pr_item_value',
    'pr_quantity',
    'log_pr_quantity',
    'unit_price',
    'log_unit_price',
    'planned_delivery_days',
    'pr_to_delivery_horizon',
    'release_state_code',
    'release_strategy_code',
    'release_group_code',
    'has_release_block',
    'processing_status_code',
    'deletion_flag',
    'has_assigned_vendor',
    'account_assignment_code',
    'item_category_code',
    'purchasing_group_freq',
    'plant_freq',
    'material_group_freq',
    'material_type_freq',
    'vendor_freq',
    'company_code_freq',
    'pr_creation_month',
    'pr_creation_dayofweek',
    'is_weekend_pr',
    'is_quarter_end_pr',
    'delivery_urgency_ratio',
    'high_value_pr_flag',
    'plant_avg_cycle_time',
    'matkl_avg_cycle_time',
    'ekgrp_avg_cycle_time'
]


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw SAP PR data across HANA_P2 and HANA_P3 and constructs 32 domain features
    for cycle time regression modeling.
    """
    df_clean = df.copy()

    # Clean text columns to avoid "nan" or "None" literal string artifacts
    str_cols = [
        'pr_number', 'pr_item', 'doc_type', 'processing_status', 'deletion_indicator',
        'release_indicator', 'release_group', 'release_strategy', 'purchasing_group',
        'requisitioner', 'po_number', 'po_item', 'material_id', 'material_desc',
        'material_type', 'material_group', 'vendor_id', 'vendor_name', 'plant',
        'plant_name', 'plant_city', 'company_code', 'company_name', 'company_city',
        'company_country', 'company_currency', 'unit_of_measure', 'currency'
    ]
    for sc in str_cols:
        if sc in df_clean.columns:
            s_series = df_clean[sc].astype(str).str.strip()
            # Replace invalid null representations with empty string
            df_clean[sc] = s_series.replace(['nan', 'None', 'NAN', 'NONE', 'NaT', 'nat', '<NA>', 'null', 'NULL'], '')

    # 1. Clean Dates
    df_clean['pr_date'] = pd.to_datetime(df_clean['requisition_date'], errors='coerce')
    df_clean['create_date'] = pd.to_datetime(df_clean['creation_date'], errors='coerce')
    df_clean['deliv_date'] = pd.to_datetime(df_clean['delivery_date'], errors='coerce')
    df_clean['po_dt'] = pd.to_datetime(df_clean['po_date'], errors='coerce')

    # Fallback to creation date if requisition date is missing
    df_clean['pr_date'] = df_clean['pr_date'].fillna(df_clean['create_date']).fillna(pd.to_datetime('2024-04-01'))
    df_clean['deliv_date'] = df_clean['deliv_date'].fillna(df_clean['pr_date'] + pd.Timedelta(days=30))

    # Reference audit date for open PR age calculation
    max_audit_date = df_clean['pr_date'].max()
    if pd.isna(max_audit_date):
        max_audit_date = pd.to_datetime('2025-06-30')
    df_clean['pr_age_days'] = (max_audit_date - df_clean['pr_date']).dt.days.clip(lower=0)

    # 2. Numerical Quantities and Values
    df_clean['pr_quantity'] = pd.to_numeric(df_clean['quantity'], errors='coerce').fillna(1.0).clip(lower=0.01)
    df_clean['raw_unit_price'] = pd.to_numeric(df_clean['unit_price'], errors='coerce').fillna(100.0).clip(lower=0.01)
    df_clean['raw_price_unit'] = pd.to_numeric(df_clean['price_unit'], errors='coerce').replace(0, 1).fillna(1.0).clip(lower=1.0)
    
    df_clean['unit_price'] = df_clean['raw_unit_price'] / df_clean['raw_price_unit']
    df_clean['pr_item_value'] = df_clean['pr_quantity'] * df_clean['unit_price']

    # Log transforms for skewed monetary & quantity distributions
    df_clean['log_pr_item_value'] = np.log1p(np.maximum(0.0, df_clean['pr_item_value']))
    df_clean['log_pr_quantity'] = np.log1p(np.maximum(0.0, df_clean['pr_quantity']))
    df_clean['log_unit_price'] = np.log1p(np.maximum(0.0, df_clean['unit_price']))

    # 3. Delivery Horizons and Planned Lead Times
    df_clean['planned_delivery_days'] = pd.to_numeric(df_clean['planned_delivery_time'], errors='coerce').fillna(30.0).clip(lower=1.0, upper=365.0)
    df_clean['pr_to_delivery_horizon'] = (df_clean['deliv_date'] - df_clean['pr_date']).dt.days.fillna(30.0).clip(lower=-30.0, upper=365.0)
    df_clean['delivery_urgency_ratio'] = np.maximum(1.0, df_clean['pr_to_delivery_horizon']) / (df_clean['planned_delivery_days'] + 1.0)

    # 4. Status, Deletion, and Release Indicators
    frg_str = df_clean['release_indicator'].str.upper()
    df_clean['release_state_code'] = np.where(frg_str == '2', 1.0, np.where(frg_str == 'X', 0.0, 0.5))
    df_clean['has_release_block'] = np.where(frg_str != '2', 1, 0)

    statu_str = df_clean['processing_status'].str.upper()
    df_clean['processing_status_code'] = np.where(statu_str == 'B', 1.0, np.where(statu_str == 'A', 0.5, 0.0))

    loekz_str = df_clean['deletion_indicator'].str.upper()
    df_clean['deletion_flag'] = np.where(loekz_str == 'X', 1, 0)

    # Vendor Assignment
    v_str = df_clean['vendor_id'].str.upper()
    df_clean['has_assigned_vendor'] = np.where(v_str != '', 1, 0)

    # Categorical Encoded Values
    df_clean['release_strategy_code'] = df_clean['release_strategy'].apply(lambda x: hash(str(x)) % 100 / 100.0)
    df_clean['release_group_code'] = df_clean['release_group'].apply(lambda x: hash(str(x)) % 100 / 100.0)
    df_clean['account_assignment_code'] = df_clean['acct_assignment_cat'].apply(lambda x: hash(str(x)) % 100 / 100.0)
    df_clean['item_category_code'] = pd.to_numeric(df_clean['item_category'], errors='coerce').fillna(0.0)

    # Frequency encodings
    for col, new_col in [
        ('purchasing_group', 'purchasing_group_freq'),
        ('plant', 'plant_freq'),
        ('material_group', 'material_group_freq'),
        ('material_type', 'material_type_freq'),
        ('vendor_id', 'vendor_freq'),
        ('company_code', 'company_code_freq')
    ]:
        freq = df_clean[col].value_counts(normalize=True)
        df_clean[new_col] = df_clean[col].map(freq).fillna(0.0)

    # Temporal features
    df_clean['pr_creation_month'] = df_clean['pr_date'].dt.month.fillna(6).astype(float)
    df_clean['pr_creation_dayofweek'] = df_clean['pr_date'].dt.dayofweek.fillna(2).astype(float)
    df_clean['is_weekend_pr'] = np.where(df_clean['pr_creation_dayofweek'] >= 5, 1, 0)
    df_clean['is_quarter_end_pr'] = np.where(df_clean['pr_creation_month'].isin([3, 6, 9, 12]), 1, 0)

    # High value threshold (90th percentile)
    val_p90 = df_clean['pr_item_value'].quantile(0.90)
    df_clean['high_value_pr_flag'] = np.where(df_clean['pr_item_value'] >= val_p90, 1, 0)

    # 5. Target Variable: Actual PR to PO Cycle Time in Days
    raw_cycle_days = (df_clean['po_dt'] - df_clean['pr_date']).dt.days
    # Clean negative retro-dated entries to 0.0 (same-day PO conversion)
    df_clean['actual_cycle_time_days'] = np.where(raw_cycle_days < 0, 0.0, raw_cycle_days)
    # If PO date or PO number is unpopulated (open PR), actual_cycle_time_days is NaN
    is_open = (df_clean['po_number'] == '') | (df_clean['po_dt'].isna())
    df_clean.loc[is_open, 'actual_cycle_time_days'] = np.nan

    # Historical Benchmark Cycle Times across Plants and Material Groups
    plant_means = df_clean.groupby('plant')['actual_cycle_time_days'].mean().fillna(28.0)
    matkl_means = df_clean.groupby('material_group')['actual_cycle_time_days'].mean().fillna(30.0)
    ekgrp_means = df_clean.groupby('purchasing_group')['actual_cycle_time_days'].mean().fillna(25.0)

    df_clean['plant_avg_cycle_time'] = df_clean['plant'].map(plant_means).fillna(28.0).clip(lower=1.0, upper=180.0)
    df_clean['matkl_avg_cycle_time'] = df_clean['material_group'].map(matkl_means).fillna(30.0).clip(lower=1.0, upper=180.0)
    df_clean['ekgrp_avg_cycle_time'] = df_clean['purchasing_group'].map(ekgrp_means).fillna(25.0).clip(lower=1.0, upper=180.0)

    return df_clean


def prepare_regression_train_test(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> dict:
    """
    Prepares regression training and test feature matrices using RobustScaler.
    Trains and tests on closed records where ground-truth cycle time is known.
    """
    closed_mask = df['actual_cycle_time_days'].notna()
    df_closed = df[closed_mask].copy()

    X = df_closed[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df_closed['actual_cycle_time_days'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Also prepare all records scaled for full production inference
    X_all = df[FEATURE_COLUMNS].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_all_scaled = scaler.transform(X_all)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_all_scaled': X_all_scaled,
        'scaler': scaler,
        'feature_cols': FEATURE_COLUMNS,
        'n_closed': len(df_closed),
        'n_open': len(df) - len(df_closed)
    }
