"""
Preprocessing and Feature Engineering Module - Use Case 7: Production Order Delay Prediction
Cleans raw SAP production order records from AFKO, AUFK, AFPO, AFRU, MAKT, MARA, MARC.
Engineers 30 domain features capturing schedule compression, release lead times,
order quantity strain, shop floor confirmation yield, scrap severity, and operational risk.
Strictly handles missing values: genuine blanks are preserved where data is absent.
Prepares unsupervised input matrix (X) standardized using RobustScaler.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


FEATURE_COLUMNS = [
    'planned_duration_days',
    'creation_to_start_lead_days',
    'release_delay_days',
    'scheduled_vs_basic_slack_days',
    'total_order_qty_log',
    'planned_scrap_ratio',
    'delivered_qty_ratio',
    'unfulfilled_qty_log',
    'confirmed_yield_ratio',
    'confirmation_scrap_rate',
    'confirmation_count_log',
    'confirmation_duration_days',
    'confirmation_intensity',
    'is_zero_confirmation',
    'delivery_completed_flag_numeric',
    'has_batch_flag',
    'planned_deliv_lead_days',
    'plant_safety_stock_log',
    'net_weight_log',
    'order_month',
    'order_quarter',
    'schedule_compression_index',
    'scrap_severity_index',
    'lead_time_strain',
    'order_type_encoded',
    'plant_encoded',
    'mrp_controller_encoded',
    'supervisor_encoded',
    'material_type_encoded',
    'composite_delay_signal'
]


def clean_and_engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw production order data and engineers 30 domain features
    for unsupervised production order delay clustering and risk scoring.
    Handles missing values: if absent, leaves genuinely blank.
    """
    df = df_raw.copy()

    # 1. Clean numeric quantities and confirmations for ML feature calculation
    numeric_cols = [
        'total_order_qty', 'planned_scrap_qty', 'header_confirmed_qty',
        'item_target_qty', 'item_delivered_qty', 'item_scrap_qty',
        'total_conf_yield', 'total_conf_scrap', 'total_confirmations',
        'net_weight', 'planned_deliv_lead_days', 'plant_safety_stock'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 2. Clean text fields: If present keep value, if absent/null leave blank "" (NO 'UNKNOWN')
    text_cols = [
        'order_number', 'order_type', 'plant', 'order_description', 'company_code',
        'material_number', 'material_name', 'mrp_controller', 'production_supervisor',
        'material_type', 'material_group', 'base_uom', 'item_uom', 'batch_number', 'storage_loc'
    ]
    for str_col in text_cols:
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(str).str.strip()
            df.loc[df[str_col].isin(['None', 'nan', 'UNKNOWN', 'NaN', 'null', 'NoneType', 'NaT']), str_col] = ''

    # 3. Date conversions
    df['planned_finish_dt'] = pd.to_datetime(df['planned_finish_date'], errors='coerce')
    df['planned_start_dt'] = pd.to_datetime(df['planned_start_date'], errors='coerce')
    df['order_creation_dt'] = pd.to_datetime(df['order_creation_date'], errors='coerce')
    df['actual_release_dt'] = pd.to_datetime(df['actual_release_date'], errors='coerce')
    df['scheduled_finish_dt'] = pd.to_datetime(df['scheduled_finish_date'], errors='coerce')
    df['scheduled_start_dt'] = pd.to_datetime(df['scheduled_start_date'], errors='coerce')
    df['technical_completion_dt'] = pd.to_datetime(df['technical_completion_date'], errors='coerce')
    df['actual_finish_dt'] = pd.to_datetime(df['actual_finish_date'], errors='coerce')
    df['actual_delivery_dt'] = pd.to_datetime(df['actual_delivery_date'], errors='coerce')
    df['last_conf_dt'] = pd.to_datetime(df['last_conf_date'], errors='coerce')
    df['min_conf_start_dt'] = pd.to_datetime(df['min_conf_start'], errors='coerce')
    df['max_conf_end_dt'] = pd.to_datetime(df['max_conf_end'], errors='coerce')

    # Effective start and finish dates for lead-time calculation
    eff_start = df['planned_start_dt'].combine_first(df['order_creation_dt']).combine_first(df['planned_finish_dt'])
    eff_finish = df['planned_finish_dt'].combine_first(eff_start)

    # 4. Schedule & Lead Time Features
    # Planned duration
    df['planned_duration_days'] = (eff_finish - eff_start).dt.days.fillna(1.0)
    df['planned_duration_days'] = np.clip(df['planned_duration_days'], 0.0, 365.0)

    # Lead time between order creation and planned start
    df['creation_to_start_lead_days'] = (eff_start - df['order_creation_dt']).dt.days.fillna(0.0)
    df['creation_to_start_lead_days'] = np.clip(df['creation_to_start_lead_days'], -30.0, 180.0)

    # Release lag: delay between release and planned start
    df['release_delay_days'] = (df['actual_release_dt'] - eff_start).dt.days.fillna(0.0)
    df['release_delay_days'] = np.clip(df['release_delay_days'], -30.0, 90.0)

    # Scheduled vs Basic finish buffer (slack)
    df['scheduled_vs_basic_slack_days'] = (df['planned_finish_dt'] - df['scheduled_finish_dt']).dt.days.fillna(0.0)
    df['scheduled_vs_basic_slack_days'] = np.clip(df['scheduled_vs_basic_slack_days'], -60.0, 60.0)

    # 5. Quantity and Scrap Ratios
    df['total_order_qty_log'] = np.log1p(np.maximum(0.0, df['total_order_qty']))
    df['planned_scrap_ratio'] = df['planned_scrap_qty'] / (df['total_order_qty'] + 1e-4)
    df['planned_scrap_ratio'] = np.clip(df['planned_scrap_ratio'], 0.0, 1.0)

    # Item target vs delivered
    item_target = np.where(df['item_target_qty'] > 0, df['item_target_qty'], df['total_order_qty'])
    df['delivered_qty_ratio'] = df['item_delivered_qty'] / (item_target + 1e-4)
    df['delivered_qty_ratio'] = np.clip(df['delivered_qty_ratio'], 0.0, 2.0)

    unfulfilled = np.maximum(0.0, item_target - df['item_delivered_qty'])
    df['unfulfilled_qty_log'] = np.log1p(unfulfilled)

    # 6. Shop Floor Confirmations & Yield
    df['confirmed_yield_ratio'] = df['total_conf_yield'] / (df['total_order_qty'] + 1e-4)
    df['confirmed_yield_ratio'] = np.clip(df['confirmed_yield_ratio'], 0.0, 2.0)

    tot_conf = df['total_conf_yield'] + df['total_conf_scrap']
    df['confirmation_scrap_rate'] = df['total_conf_scrap'] / (tot_conf + 1e-4)
    df['confirmation_scrap_rate'] = np.clip(df['confirmation_scrap_rate'], 0.0, 1.0)

    df['confirmation_count_log'] = np.log1p(np.maximum(0.0, df['total_confirmations']))

    conf_dur = (df['max_conf_end_dt'] - df['min_conf_start_dt']).dt.days.fillna(0.0)
    df['confirmation_duration_days'] = np.clip(conf_dur, 0.0, 180.0)

    df['confirmation_intensity'] = df['total_confirmations'] / (df['planned_duration_days'] + 1.0)
    df['confirmation_intensity'] = np.clip(df['confirmation_intensity'], 0.0, 50.0)

    df['is_zero_confirmation'] = (df['total_confirmations'] == 0.0).astype(float)
    df['delivery_completed_flag_numeric'] = (df['delivery_completed_flag'] == 'X').astype(float)
    df['has_batch_flag'] = (df['batch_number'] != '').astype(float)

    # 7. Material Attributes & Safety Stocks
    df['planned_deliv_lead_days'] = np.clip(df['planned_deliv_lead_days'], 0.0, 180.0)
    df['plant_safety_stock_log'] = np.log1p(np.maximum(0.0, df['plant_safety_stock']))
    df['net_weight_log'] = np.log1p(np.maximum(0.0, df['net_weight']))

    # Seasonality
    df['order_month'] = eff_start.dt.month.fillna(6).astype(float)
    df['order_quarter'] = eff_start.dt.quarter.fillna(2).astype(float)

    # 8. Stress & Risk Indexes
    df['schedule_compression_index'] = df['total_order_qty_log'] / (df['planned_duration_days'] + 1.0)
    df['scrap_severity_index'] = (df['planned_scrap_ratio'] * 0.40) + (df['confirmation_scrap_rate'] * 0.60)
    df['lead_time_strain'] = (df['creation_to_start_lead_days'] / (df['planned_duration_days'] + 1.0)).clip(-5.0, 10.0)

    # 9. Categorical Frequency Encodings
    for cat_col, enc_col in [
        ('order_type', 'order_type_encoded'),
        ('plant', 'plant_encoded'),
        ('mrp_controller', 'mrp_controller_encoded'),
        ('production_supervisor', 'supervisor_encoded'),
        ('material_type', 'material_type_encoded')
    ]:
        freq_map = df[cat_col].value_counts(normalize=True).to_dict()
        df[enc_col] = df[cat_col].map(freq_map).fillna(0.0).astype(float)

    # 10. Composite Delay Risk Signal (Unsupervised Domain Heuristic)
    df['composite_delay_signal'] = (
        (np.clip(df['release_delay_days'] / 30.0, -1.0, 3.0) * 0.25) +
        (np.clip(df['scrap_severity_index'], 0.0, 1.0) * 0.25) +
        (np.clip((1.0 - df['delivered_qty_ratio']), 0.0, 1.0) * 0.25) +
        (df['is_zero_confirmation'] * 0.15) +
        (np.clip(-df['scheduled_vs_basic_slack_days'] / 30.0, 0.0, 2.0) * 0.10)
    )

    # 11. Ground Truth Delay Benchmark Definition (Post-Hoc Validation Only)
    df['actual_completion_dt'] = (
        df['actual_finish_dt']
        .combine_first(df['actual_delivery_dt'])
        .combine_first(df['technical_completion_dt'])
        .combine_first(df['last_conf_dt'])
    )

    df['actual_delay_days'] = (df['actual_completion_dt'] - df['planned_finish_dt']).dt.days
    # Delayed if actual completion > planned finish date
    df['is_delayed_exception'] = (df['actual_delay_days'] > 0.0).astype(int)

    # For orders without actual completion recorded: check if planned finish has passed and unfulfilled
    ref_now = pd.to_datetime('2025-09-22')
    is_open_past_due = (df['actual_completion_dt'].isna()) & (df['planned_finish_dt'] < ref_now) & (df['delivery_completed_flag_numeric'] == 0.0)
    df.loc[is_open_past_due, 'is_delayed_exception'] = 1

    # Missing value formatting for dates in export
    df['actual_completion_formatted'] = df['actual_completion_dt'].dt.strftime('%Y-%m-%d').fillna('')
    df['actual_delay_days_display'] = np.where(df['actual_delay_days'].isna(), '', df['actual_delay_days'].round(1).astype(str))

    return df


def prepare_train_test_data(
    df_featured: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> dict:
    """
    Extracts 30-feature matrix X, applies RobustScaler, and partitions into train/test splits.
    Note: y_train and y_test are extracted purely for post-hoc benchmark evaluation;
    all 5 ML models train strictly unsupervised on X without any labels.
    """
    X = df_featured[FEATURE_COLUMNS].values
    y = df_featured['is_delayed_exception'].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_featured.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'idx_train': idx_train,
        'idx_test': idx_test,
        'feature_cols': FEATURE_COLUMNS,
        'scaler': scaler
    }
