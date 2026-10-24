"""
Preprocessing and Feature Engineering Module - Use Case 6: Slow Moving Inventory Prediction
Cleans raw SAP material master, valuation, and transaction records across HANA_P2 and HANA_P3.
Implements robust missing value handling: text fields left blank ("") when absent, numeric values properly imputed.
Engineers 32 domain features capturing inventory inactivity, turnover velocity,
days of supply, capital exposure, safety stock variance, and procurement imbalance.
Prepares unsupervised input matrix (X) and standardizes using RobustScaler.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


TEXT_METADATA_COLS = [
    'source_schema', 'matnr', 'plant', 'material_name', 'plant_name',
    'material_type', 'material_group', 'base_uom', 'price_control',
    'valuation_class', 'abc_indicator', 'mrp_type', 'mrp_controller',
    'lot_size_key', 'procurement_type', 'material_created_on',
    'last_change_date', 'last_issue_date', 'last_receipt_date', 'last_movement_date'
]

NUMERIC_COLS = [
    'current_stock_qty', 'current_stock_val', 'moving_avg_price', 'standard_price',
    'prev_period_stock_qty', 'prev_period_stock_val', 'prev_year_stock_qty', 'prev_year_stock_val',
    'net_weight', 'gross_weight', 'planned_deliv_days', 'reorder_point', 'safety_stock',
    'total_issue_qty', 'total_issue_val', 'total_receipt_qty', 'total_receipt_val',
    'issue_txn_cnt', 'receipt_txn_cnt', 'total_movement_cnt'
]

FEATURE_COLUMNS = [
    'days_since_last_issue',
    'days_since_last_movement',
    'days_since_creation',
    'is_inactive_90d',
    'is_inactive_180d',
    'is_zero_issue_window',
    'daily_consumption_rate',
    'days_of_supply_forward',
    'inventory_turnover_ratio',
    'annualized_turnover_rate',
    'stock_value_log',
    'current_stock_qty_log',
    'unit_valuation_price',
    'price_variance_ratio',
    'high_value_stagnant_flag',
    'receipt_to_issue_ratio',
    'net_accumulation_qty',
    'stock_drift_previous_period',
    'stock_drift_previous_year',
    'stock_to_safety_ratio',
    'excess_safety_stock_qty',
    'excess_safety_stock_val',
    'reorder_point_gap',
    'issue_transaction_density',
    'inactivity_rate_window',
    'obsolescence_risk_index',
    'capital_lockup_strain',
    'abc_numeric',
    'planned_deliv_days',
    'net_weight_log',
    'mrp_type_encoded',
    'material_type_encoded'
]


def clean_and_engineer_features(df_raw: pd.DataFrame, ref_date_str: str = "2026-07-31") -> pd.DataFrame:
    """
    Cleans raw inventory data and engineers 32 domain features for unsupervised slow-moving/obsolescence detection.
    Missing values in text/categorical columns are cleanly formatted as blank ("").
    Numeric columns are converted to clean finite floats.
    """
    df = df_raw.copy()
    ref_date = pd.to_datetime(ref_date_str)

    # 1. Clean and standardize text/metadata fields (empty strings when missing)
    for col in TEXT_METADATA_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
            df[col] = df[col].replace({'None': '', 'nan': '', 'NaN': '', 'null': '', 'NULL': ''})

    # 2. Clean numeric fields
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Ensure non-negative stock and value
    df['current_stock_qty'] = np.maximum(0.0, df['current_stock_qty'])
    df['current_stock_val'] = np.maximum(0.0, df['current_stock_val'])

    # 3. Date handling and inactivity calculations
    df['last_issue_date_dt'] = pd.to_datetime(df['last_issue_date'], errors='coerce')
    df['last_movement_date_dt'] = pd.to_datetime(df['last_movement_date'], errors='coerce')
    df['material_created_on_dt'] = pd.to_datetime(df['material_created_on'], errors='coerce')

    # Days since last issue (consumption / sales)
    df['days_since_last_issue'] = (ref_date - df['last_issue_date_dt']).dt.days.fillna(365.0)
    df['days_since_last_issue'] = np.clip(df['days_since_last_issue'], 0.0, 365.0)

    # Days since last movement (any GR/GI)
    df['days_since_last_movement'] = (ref_date - df['last_movement_date_dt']).dt.days.fillna(365.0)
    df['days_since_last_movement'] = np.clip(df['days_since_last_movement'], 0.0, 365.0)

    # Days since creation (material relationship tenure)
    df['days_since_creation'] = (ref_date - df['material_created_on_dt']).dt.days.fillna(730.0)
    df['days_since_creation'] = np.maximum(1.0, df['days_since_creation'])

    # 4. Inactivity binary indicators
    df['is_inactive_90d'] = (df['days_since_last_issue'] >= 90.0).astype(float)
    df['is_inactive_180d'] = (df['days_since_last_issue'] >= 180.0).astype(float)
    df['is_zero_issue_window'] = (df['issue_txn_cnt'] == 0.0).astype(float)

    # 5. Consumption and turnover velocity (observation window: 365 days)
    observation_days = 365.0
    df['daily_consumption_rate'] = df['total_issue_qty'] / observation_days

    # Days of forward supply
    df['days_of_supply_forward'] = df['current_stock_qty'] / (df['daily_consumption_rate'] + 1e-4)
    df['days_of_supply_forward'] = np.clip(df['days_of_supply_forward'], 0.0, 1000.0)

    # Inventory turnover ratio (Issue Qty / Stock Qty)
    df['inventory_turnover_ratio'] = df['total_issue_qty'] / (df['current_stock_qty'] + 1e-4)
    df['annualized_turnover_rate'] = np.clip(df['inventory_turnover_ratio'], 0.0, 50.0)

    # 6. Financial exposure and pricing
    df['stock_value_log'] = np.log1p(df['current_stock_val'])
    df['current_stock_qty_log'] = np.log1p(df['current_stock_qty'])
    df['unit_valuation_price'] = df['current_stock_val'] / (df['current_stock_qty'] + 1e-4)
    
    price_diff = np.abs(df['moving_avg_price'] - df['standard_price'])
    price_base = np.maximum(df['standard_price'], df['moving_avg_price'])
    df['price_variance_ratio'] = price_diff / (price_base + 1e-4)
    df['price_variance_ratio'] = np.clip(df['price_variance_ratio'], 0.0, 5.0)

    # High-value stagnant capital flag (>500k INR & inactive >= 90d)
    df['high_value_stagnant_flag'] = (
        (df['current_stock_val'] > 500000.0) & (df['days_since_last_issue'] >= 90.0)
    ).astype(float)

    # 7. Inflow vs Outflow imbalances
    df['receipt_to_issue_ratio'] = df['total_receipt_qty'] / (df['total_issue_qty'] + 1e-4)
    df['receipt_to_issue_ratio'] = np.clip(df['receipt_to_issue_ratio'], 0.0, 20.0)
    df['net_accumulation_qty'] = df['total_receipt_qty'] - df['total_issue_qty']

    # Historical stock drifts
    df['stock_drift_previous_period'] = (
        (df['current_stock_qty'] - df['prev_period_stock_qty']) / (df['prev_period_stock_qty'] + 1e-4)
    )
    df['stock_drift_previous_period'] = np.clip(df['stock_drift_previous_period'], -5.0, 5.0)

    df['stock_drift_previous_year'] = (
        (df['current_stock_qty'] - df['prev_year_stock_qty']) / (df['prev_year_stock_qty'] + 1e-4)
    )
    df['stock_drift_previous_year'] = np.clip(df['stock_drift_previous_year'], -5.0, 5.0)

    # 8. Safety stock and reorder point buffers
    df['stock_to_safety_ratio'] = df['current_stock_qty'] / (df['safety_stock'] + 1e-4)
    df['stock_to_safety_ratio'] = np.clip(df['stock_to_safety_ratio'], 0.0, 50.0)

    df['excess_safety_stock_qty'] = np.maximum(0.0, df['current_stock_qty'] - df['safety_stock'])
    df['excess_safety_stock_val'] = df['excess_safety_stock_qty'] * df['unit_valuation_price']

    df['reorder_point_gap'] = df['current_stock_qty'] - df['reorder_point']

    # 9. Transaction frequency and inactivity index
    df['issue_transaction_density'] = df['issue_txn_cnt'] / (df['total_movement_cnt'] + 1e-4)
    df['inactivity_rate_window'] = df['days_since_last_issue'] / 365.0

    # Composite Obsolescence Risk Index
    df['obsolescence_risk_index'] = (
        (df['inactivity_rate_window'] * 0.40) +
        (np.clip(df['days_of_supply_forward'] / 365.0, 0.0, 2.0) / 2.0 * 0.30) +
        (df['is_zero_issue_window'] * 0.30)
    )

    # Capital Lockup Strain (combines financial size with dormancy)
    df['capital_lockup_strain'] = (df['stock_value_log'] / 20.0) * df['obsolescence_risk_index']

    # 10. Categorical encodings
    abc_map = {'A': 3.0, 'B': 2.0, 'C': 1.0}
    df['abc_numeric'] = df['abc_indicator'].map(abc_map).fillna(0.0)

    df['planned_deliv_days'] = np.clip(df['planned_deliv_days'], 0.0, 180.0)
    df['net_weight_log'] = np.log1p(np.maximum(0.0, df['net_weight']))

    # MRP Type encoding
    mrp_counts = df['mrp_type'].value_counts(normalize=True).to_dict()
    df['mrp_type_encoded'] = df['mrp_type'].map(mrp_counts).fillna(0.0)

    # Material Type encoding
    mtart_counts = df['material_type'].value_counts(normalize=True).to_dict()
    df['material_type_encoded'] = df['material_type'].map(mtart_counts).fillna(0.0)

    # 11. Post-hoc Ground Truth Benchmark Definition (MC46/MC50 standard)
    df['is_slow_moving_exception'] = (
        (df['days_since_last_issue'] >= 90.0) &
        ((df['issue_txn_cnt'] == 0.0) | (df['inventory_turnover_ratio'] < 0.10) | (df['days_of_supply_forward'] > 365.0))
    ).astype(int)

    # Drop temporary datetime parsing columns
    drop_temp = ['last_issue_date_dt', 'last_movement_date_dt', 'material_created_on_dt']
    df = df.drop(columns=[c for c in drop_temp if c in df.columns])

    return df


def prepare_train_test_data(
    df_featured: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> dict:
    """
    Extracts 32-feature matrix X, applies RobustScaler, and partitions into train/test splits.
    Note: y_train and y_test are extracted solely for post-hoc validation; models train 100% unsupervised.
    """
    X = df_featured[FEATURE_COLUMNS].values
    y = df_featured['is_slow_moving_exception'].values

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Robust scaling (resilient to heavy financial outliers in inventory values)
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
        'scaler': scaler,
        'feature_cols': FEATURE_COLUMNS
    }
