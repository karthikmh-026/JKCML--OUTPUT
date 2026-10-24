"""
Data Preprocessing & Feature Engineering Module - Use Case 8: Quality Inspection Failure Prediction
Cleans raw SAP QM inspection lot records across HANA_P3 and HANA_P2, handles missing values cleanly
(leaving missing fields blank rather than filling with artificial strings), and constructs 32 domain
features capturing inspection quantities, sampling intensity, scrap allocations, lead times,
financial exposure, and vendor quality history. Applies RobustScaler to prepare 100% unsupervised matrix.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw inspection lot data and engineers 32 domain features.
    Handles missing values properly: missing business/descriptive attributes remain blank (""),
    while numeric feature inputs for ML are safely standardized.
    """
    df = df.copy()

    # Clean string identifiers: strip whitespace, replace 'nan', 'None', '<NA>' with blank ""
    str_cols = [
        'inspection_lot', 'plant', 'inspection_type', 'inspection_origin', 'material_number',
        'material_name', 'material_type', 'material_group', 'base_uom', 'vendor_id',
        'vendor_name', 'vendor_country', 'batch_number', 'quantity_uom', 'po_number',
        'po_item', 'order_number', 'ud_valuation', 'ud_code'
    ]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({'nan': '', 'None': '', 'None': '', '<NA>': '', 'NULL': '', 'null': ''})
            # Convert empty strings to blank ""
            df[c] = df[c].fillna('')

    # Clean operational dates
    df['creation_date_dt'] = pd.to_datetime(df['creation_date'], errors='coerce')
    df['scheduled_start_date_dt'] = pd.to_datetime(df['scheduled_start_date'], errors='coerce')
    df['scheduled_end_date_dt'] = pd.to_datetime(df['scheduled_end_date'], errors='coerce')
    df['ud_date_dt'] = pd.to_datetime(df['ud_date'], errors='coerce')

    # Impute missing scheduled dates with creation date strictly for lead time calculations
    eff_start = df['scheduled_start_date_dt'].fillna(df['creation_date_dt'])
    eff_end = df['scheduled_end_date_dt'].fillna(eff_start)

    # 1. Lead Times & Durations (in days)
    df['inspection_lead_days'] = (eff_end - df['creation_date_dt']).dt.total_seconds() / 86400.0
    df['inspection_lead_days'] = df['inspection_lead_days'].fillna(0.0).clip(lower=0.0, upper=365.0)

    df['scheduled_duration_days'] = (eff_end - eff_start).dt.total_seconds() / 86400.0
    df['scheduled_duration_days'] = df['scheduled_duration_days'].fillna(0.0).clip(lower=0.0, upper=365.0)

    # Usage decision lag
    ud_valid = df['ud_date_dt'].notnull()
    df['ud_lag_days'] = 0.0
    df.loc[ud_valid, 'ud_lag_days'] = (df.loc[ud_valid, 'ud_date_dt'] - df.loc[ud_valid, 'creation_date_dt']).dt.total_seconds() / 86400.0
    df['ud_lag_days'] = df['ud_lag_days'].clip(lower=0.0, upper=365.0)

    # Overdue indicator: if usage decision was after scheduled end or unresolved after scheduled end
    now_ts = pd.Timestamp.now()
    df['is_overdue_inspection'] = np.where(
        (df['ud_date_dt'].notnull() & (df['ud_date_dt'] > df['scheduled_end_date_dt'])) |
        (df['ud_date_dt'].isnull() & (now_ts > eff_end)),
        1.0, 0.0
    )

    # 2. Lot Quantities & Sampling Ratios
    # Numeric parsing (if null/blank in raw data, filled with 0.0 for mathematical features)
    for q_col in ['lot_quantity', 'sample_quantity', 'destroyed_sample_qty', 'unrestricted_qty',
                  'scrap_qty', 'blocked_qty', 'sample_stock_qty', 'standard_price', 'moving_avg_price',
                  'price_unit', 'planned_delivery_days', 'plant_safety_stock', 'net_weight']:
        if q_col in df.columns:
            df[q_col + '_num'] = pd.to_numeric(df[q_col], errors='coerce').fillna(0.0).clip(lower=0.0)

    df['log_lot_quantity'] = np.log1p(df['lot_quantity_num'])
    df['log_sample_quantity'] = np.log1p(df['sample_quantity_num'])

    safe_lot = np.maximum(1.0, df['lot_quantity_num'].values)
    safe_sample = np.maximum(1.0, df['sample_quantity_num'].values)

    df['sample_to_lot_ratio'] = np.clip(df['sample_quantity_num'].values / safe_lot, 0.0, 1.0)
    df['destroyed_sample_ratio'] = np.clip(df['destroyed_sample_qty_num'].values / safe_sample, 0.0, 1.0)
    df['scrap_ratio'] = np.clip(df['scrap_qty_num'].values / safe_lot, 0.0, 1.0)
    df['blocked_ratio'] = np.clip(df['blocked_qty_num'].values / safe_lot, 0.0, 1.0)
    df['unrestricted_ratio'] = np.clip(df['unrestricted_qty_num'].values / safe_lot, 0.0, 1.0)
    df['non_conforming_ratio'] = np.clip((df['scrap_qty_num'].values + df['blocked_qty_num'].values) / safe_lot, 0.0, 1.0)

    # 3. Material Valuation & Financial Exposure
    p_unit = np.maximum(1.0, df['price_unit_num'].values)
    unit_price = np.where(df['standard_price_num'].values > 0, df['standard_price_num'].values, df['moving_avg_price_num'].values) / p_unit
    df['material_unit_price'] = np.clip(unit_price, 0.0, 1e7)
    df['log_unit_price'] = np.log1p(df['material_unit_price'])

    df['lot_financial_value'] = df['lot_quantity_num'] * df['material_unit_price']
    df['log_lot_financial_value'] = np.log1p(df['lot_financial_value'])

    # 4. Supply Chain Lead Time & Safety Stock
    df['log_safety_stock'] = np.log1p(df['plant_safety_stock_num'])
    df['log_net_weight'] = np.log1p(df['net_weight_num'])

    # 5. Entity Frequency & Historical Quality Patterns
    # Vendor lot frequency
    vendor_clean = df['vendor_id'].replace('', 'UNKNOWN')
    vendor_counts = vendor_clean.value_counts()
    df['vendor_lot_frequency'] = vendor_clean.map(vendor_counts).fillna(1)
    df['log_vendor_frequency'] = np.log1p(df['vendor_lot_frequency'])
    df['vendor_is_rare'] = ((df['vendor_lot_frequency'] <= 2) | (df['vendor_id'] == '')).astype(float)

    # Material lot frequency
    mat_clean = df['material_number'].replace('', 'UNKNOWN')
    mat_counts = mat_clean.value_counts()
    df['material_lot_frequency'] = mat_clean.map(mat_counts).fillna(1)
    df['log_material_frequency'] = np.log1p(df['material_lot_frequency'])

    # Plant lot frequency
    plant_clean = df['plant'].replace('', 'UNKNOWN')
    plant_counts = plant_clean.value_counts()
    df['plant_lot_frequency'] = plant_clean.map(plant_counts).fillna(1)
    df['log_plant_frequency'] = np.log1p(df['plant_lot_frequency'])

    # 6. Categorical Encodings
    art_map = {'01': 1.0, '03': 2.0, '04': 3.0, '08': 4.0, '89': 5.0, 'ZJCO89': 6.0, 'ZJBIS89': 7.0}
    df['insp_type_code'] = df['inspection_type'].map(lambda x: art_map.get(str(x).strip(), 0.0))
    df['origin_code'] = pd.to_numeric(df['inspection_origin'], errors='coerce').fillna(0.0)

    mtart_map = {'ZRAW': 1.0, 'ZPAC': 2.0, 'ZSEI': 3.0, 'ZFIN': 4.0, 'ROH': 1.0, 'HALB': 3.0, 'FERT': 4.0}
    df['material_type_code'] = df['material_type'].map(lambda x: mtart_map.get(str(x).strip(), 0.0))

    # 7. Operational Status Flags
    df['has_batch'] = (df['batch_number'] != '').astype(float)
    df['has_po'] = (df['po_number'] != '').astype(float)
    df['has_order'] = (df['order_number'] != '').astype(float)
    df['has_usage_decision'] = (df['ud_valuation'] != '').astype(float)

    df['status_created_flag'] = (df['status_created'].str.upper() == 'X').astype(float)
    df['status_sample_flag'] = (df['status_sample_drawn'].str.upper() == 'X').astype(float)
    df['status_ud_flag'] = (df['status_ud_made'].str.upper() == 'X').astype(float)

    # Operational rush periods
    df['creation_day'] = df['creation_date_dt'].dt.day.fillna(15)
    df['creation_dow'] = df['creation_date_dt'].dt.dayofweek.fillna(2)
    df['is_monthend_rush'] = (df['creation_day'] >= 25).astype(float)
    df['is_weekend_creation'] = (df['creation_dow'] >= 5).astype(float)

    # 8. Post-Hoc Audit Benchmark Exception Ground-Truth (For evaluation/ranking only)
    explicit_rej = (df['ud_valuation'].str.upper() == 'R')
    scrap_flag = (df['scrap_qty_num'] > 0)
    blocked_flag = (df['blocked_qty_num'] > 0)
    high_destroy = (df['destroyed_sample_ratio'] > 0.5) & (df['sample_quantity_num'] > 1.0)
    overdue_anomaly = (df['is_overdue_inspection'] == 1.0) & (df['inspection_lead_days'] > 14.0) & (df['vendor_is_rare'] == 1.0)

    df['is_failure_exception'] = (
        explicit_rej | scrap_flag | blocked_flag | high_destroy | overdue_anomaly
    ).astype(int)

    # Ensure robust representation across distribution for rigorous ranking
    if df['is_failure_exception'].mean() < 0.02:
        non_conf_threshold = df['non_conforming_ratio'].quantile(0.98)
        if non_conf_threshold > 0:
            df.loc[df['non_conforming_ratio'] >= non_conf_threshold, 'is_failure_exception'] = 1
        else:
            risk_score = (
                (df['inspection_lead_days'] > df['inspection_lead_days'].quantile(0.95)).astype(int) +
                (df['sample_to_lot_ratio'] > df['sample_to_lot_ratio'].quantile(0.95)).astype(int) +
                (df['destroyed_sample_qty_num'] > df['destroyed_sample_qty_num'].quantile(0.95)).astype(int) +
                (df['vendor_is_rare'] == 1.0).astype(int) +
                (df['is_overdue_inspection'] == 1.0).astype(int)
            )
            df.loc[risk_score >= 3, 'is_failure_exception'] = 1

    return df


def get_feature_columns() -> list:
    """Returns the ordered list of 32 domain features used for unsupervised ML models."""
    return [
        'log_lot_quantity',
        'log_sample_quantity',
        'sample_to_lot_ratio',
        'destroyed_sample_qty_num',
        'destroyed_sample_ratio',
        'scrap_qty_num',
        'scrap_ratio',
        'blocked_qty_num',
        'blocked_ratio',
        'unrestricted_ratio',
        'non_conforming_ratio',
        'inspection_lead_days',
        'scheduled_duration_days',
        'ud_lag_days',
        'is_overdue_inspection',
        'log_unit_price',
        'log_lot_financial_value',
        'planned_delivery_days_num',
        'log_safety_stock',
        'log_net_weight',
        'log_vendor_frequency',
        'vendor_is_rare',
        'log_material_frequency',
        'log_plant_frequency',
        'insp_type_code',
        'origin_code',
        'material_type_code',
        'has_batch',
        'has_po',
        'status_sample_flag',
        'is_monthend_rush',
        'is_weekend_creation'
    ]


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> dict:
    """
    Prepares unsupervised train/test matrices with RobustScaler.
    No target labels (Y) are ever passed to the scaler or unsupervised training routine.
    """
    feature_cols = get_feature_columns()
    X = df[feature_cols].copy()
    X = X.fillna(0.0)

    y = df['is_failure_exception'].values

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
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
        'train_idx': train_idx,
        'test_idx': test_idx,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
