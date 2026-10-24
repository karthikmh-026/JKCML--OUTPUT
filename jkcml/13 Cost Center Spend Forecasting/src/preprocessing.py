"""
Preprocessing & Time-Series Feature Engineering Module - Use Case 13: Cost Center Spend Forecasting (Time Series)
Transforms raw SAP Cost Center Spend records into 32 domain-engineered numerical Time-Series
features. Partitions data chronologically into 2025 (Training Horizon) and 2026 (Forecast Horizon)
to forecast departmental spending and identify budget variance and overrun exceptions.
"""
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


REGEX_FREIGHT_LOGISTICS = re.compile(
    r'\b(?:FREIGHT|FRT|LOADING|UNLOADING|LDG|ULDG|FORWARDING|FORWRD|DISPATCH|TRANSPORT|RAIL|RAKE|'
    r'TRUCK|FLEET|HANDLING|WHARFAGE|DEMURRAGE|DUMP|GODOWN|WAREHOUSE|LOGISTICS)\b',
    re.IGNORECASE
)

REGEX_MAINT_REPAIR_SPARES = re.compile(
    r'\b(?:MAINT|MAINTENANCE|REPAIR|OVERHAUL|BREAKDOWN|SPARE|STORE|AMC|SERVICE\s*CHG|CONSUMABLE|'
    r'LUBRICANT|OIL|FUEL|POWER|ELECTRICITY|KILN|MILL|CRUSHER|PLANT|WORKSHOP|CALIBRATION)\b',
    re.IGNORECASE
)

REGEX_SALES_MARKETING = re.compile(
    r'\b(?:SALES|PROMOTION|PROM|INCENTIVE|COMMISSION|DEALER|TOKEN|PUTTY|ADVERTISING|BRANDING|'
    r'MARKETING|REBATE|DISCOUNT|SCHEME|GIFT|CONFERENCE|S&M)\b',
    re.IGNORECASE
)

REGEX_EMPLOYEE_PAYROLL = re.compile(
    r'\b(?:SALARY|WAGE|PAYROLL|PF|PROVIDENT|CONTRIBUTION|GRATUITY|BONUS|WELFARE|STAFF|MEDICAL|'
    r'INSURANCE|CANTEEN|STIPEND|TRAINING|RECRUITMENT|HR|HUMAN\s*RESOURCE)\b',
    re.IGNORECASE
)

REGEX_ADMIN_RENT_TRAVEL = re.compile(
    r'\b(?:RENT|LEASE|TRAVEL|TRAVELLING|HOTEL|BOARDING|LODGING|CONVEYANCE|TAXI|FLIGHT|OFFICE|'
    r'UPKEEPING|STATIONERY|PRINTING|POSTAGE|COURIER|TELEPHONE|INTERNET|LEGAL|AUDIT|CONSULTANCY|'
    r'RETAINER|SECURITY|HOUSEKEEPING|BANK\s*CHARGES|GUEST\s*HOUSE|FINANCE|ACCOUNTS|TREASURY)\b',
    re.IGNORECASE
)


def _parse_sap_amount(series: pd.Series) -> pd.Series:
    """Cleans formatted SAP amount strings into numeric floats."""
    s = series.fillna('0').astype(str).str.strip().str.replace(',', '', regex=False)
    trailing_neg = s.str.endswith('-')
    s = s.str.rstrip('-').str.strip()
    vals = pd.to_numeric(s, errors='coerce').fillna(0.0)
    vals = np.where(trailing_neg, -np.abs(vals), vals)
    return pd.Series(vals, index=series.index, dtype=float)


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw SAP Cost Center Spend records from 2025 and 2026, engineers multi-scale time-series
    lag and rolling window features, departmental spend categories, and establishes the forecast baseline.
    """
    df = df.copy()

    # 1. Clean string fields
    str_cols = [
        'company_code', 'company_name', 'company_city', 'currency',
        'doc_number', 'fiscal_year', 'line_item', 'cost_center',
        'cost_center_name', 'department_code', 'po_number', 'po_item',
        'gl_account', 'gl_short_text', 'gl_description',
        'gl_account_group', 'gl_account_type',
        'plant', 'plant_name', 'plant_city', 'debit_credit_ind',
        'line_item_text', 'vendor_code', 'vendor_name', 'invoice_ref',
        'order_id', 'order_desc'
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna('')
                .astype(str)
                .str.replace('\ufffd', ' ', regex=False)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
            )
        else:
            df[col] = ''

    # 2. Financial spend amount
    raw_amount = _parse_sap_amount(df.get('spend_amount_raw', pd.Series(0.0, index=df.index))).abs()
    # If credit note (H), adjust spend
    df['actual_spend_inr'] = np.where(df['debit_credit_ind'].str.upper() == 'H', -raw_amount, raw_amount)
    df['spend_abs_inr'] = raw_amount

    for col in ['cc_po_commitment_items', 'cc_po_distinct_orders', 'cc_po_commitment_value_inr', 'cc_pm_order_count', 'cc_pm_order_types_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).abs()
        else:
            df[col] = 0.0

    # 3. Parse Calendar & Time-Series Dates (no hardcoded fallback dates)
    post_dt = pd.to_datetime(df['posting_date'], errors='coerce')
    doc_dt = pd.to_datetime(df['document_date'], errors='coerce')
    post_dt = post_dt.fillna(doc_dt)
    doc_dt = doc_dt.fillna(post_dt)

    df['posting_date_dt'] = post_dt
    df['year_month'] = post_dt.dt.strftime('%Y-%m').fillna('')
    fy_year = pd.to_numeric(df.get('fiscal_year', 0), errors='coerce').fillna(0).astype(int)
    df['posting_year'] = post_dt.dt.year.fillna(fy_year).astype(int)
    df['posting_month'] = post_dt.dt.month.fillna(0.0).astype(float)
    df['posting_quarter'] = post_dt.dt.quarter.fillna(0.0).astype(float)
    df['posting_day_of_month'] = post_dt.dt.day.fillna(0.0).astype(float)
    df['posting_day_of_week'] = post_dt.dt.dayofweek.fillna(0.0).astype(float)
    df['is_weekend_posting'] = (df['posting_day_of_week'] >= 5).astype(int)
    df['is_month_end_window'] = (df['posting_day_of_month'] >= 25).astype(int)

    lag_days = (post_dt - doc_dt).dt.days.fillna(0.0).clip(-30.0, 365.0).astype(float)
    df['posting_lag_days'] = np.abs(lag_days)

    safe_month = np.maximum(1.0, df['posting_month'])
    df['month_sin'] = np.round(np.sin(2.0 * np.pi * safe_month / 12.0), 5)
    df['month_cos'] = np.round(np.cos(2.0 * np.pi * safe_month / 12.0), 5)

    # 4. Departmental Spend Category Classification
    combined_spend_text = (
        df['gl_description'] + " " + 
        df['gl_short_text'] + " " + 
        df['cost_center_name'] + " " + 
        df['order_desc'] + " " + 
        df['line_item_text']
    ).str.upper()

    gl_clean = df['gl_account'].str.lstrip('0')

    dept_conds = [
        gl_clean.str.startswith(('3141', '3121', '3131')) | combined_spend_text.str.contains(r'\b(?:RAW|MINING|LIMESTONE|SELENITE|MATERIAL|GYPSUM|COAL|PETCOKE|PRODUCTION)\b', regex=True),
        gl_clean.str.startswith(('3323',)) | combined_spend_text.str.contains(REGEX_FREIGHT_LOGISTICS, na=False),
        gl_clean.str.startswith(('3321', '3322')) | combined_spend_text.str.contains(REGEX_SALES_MARKETING, na=False),
        gl_clean.str.startswith(('320', '321')) | combined_spend_text.str.contains(REGEX_EMPLOYEE_PAYROLL, na=False),
        gl_clean.str.startswith(('3310', '3311', '3312', '3316', '3300')) | combined_spend_text.str.contains(REGEX_ADMIN_RENT_TRAVEL, na=False),
        gl_clean.str.startswith(('3317', '312', '313')) | combined_spend_text.str.contains(REGEX_MAINT_REPAIR_SPARES, na=False)
    ]
    dept_choices = [
        'Raw Materials & Production Operations',
        'Logistics, Freight & Dispatch Operations',
        'Sales, Marketing & Dealer Incentives',
        'Employee Payroll, PF & Staff Welfare',
        'Corporate Admin, Travel, Rent & Overheads',
        'Plant Maintenance, Repairs & Technical Spares'
    ]
    df['department_spend_category'] = np.select(
        dept_conds,
        dept_choices,
        default='General Departmental & Plant Operations'
    )

    # 5. Chronological Multi-Month Time Series Aggregation & Lag/Rolling Features per Cost Center
    # Group by Cost Center and Month in chronological order
    cc_monthly = (
        df.groupby(['cost_center', 'year_month'], as_index=False)
        .agg(
            cc_monthly_actual_spend_inr=('actual_spend_inr', 'sum'),
            cc_monthly_tx_count=('actual_spend_inr', 'count'),
            cc_monthly_mean_tx_inr=('actual_spend_inr', 'mean'),
            cc_monthly_max_tx_inr=('actual_spend_inr', 'max')
        )
        .sort_values(['cost_center', 'year_month'])
        .reset_index(drop=True)
    )

    # Compute Time Series Lags per Cost Center
    cc_monthly['lag_1_month_spend_inr'] = cc_monthly.groupby('cost_center')['cc_monthly_actual_spend_inr'].shift(1)
    cc_monthly['lag_2_month_spend_inr'] = cc_monthly.groupby('cost_center')['cc_monthly_actual_spend_inr'].shift(2)
    cc_monthly['lag_3_month_spend_inr'] = cc_monthly.groupby('cost_center')['cc_monthly_actual_spend_inr'].shift(3)

    # Backfill earliest periods with current spend
    cc_monthly['lag_1_month_spend_inr'] = cc_monthly['lag_1_month_spend_inr'].fillna(cc_monthly['cc_monthly_actual_spend_inr'])
    cc_monthly['lag_2_month_spend_inr'] = cc_monthly['lag_2_month_spend_inr'].fillna(cc_monthly['lag_1_month_spend_inr'])
    cc_monthly['lag_3_month_spend_inr'] = cc_monthly['lag_3_month_spend_inr'].fillna(cc_monthly['lag_2_month_spend_inr'])

    # Rolling 3-Month Moving Average and Standard Deviation
    cc_monthly['rolling_mean_3m_spend_inr'] = (
        cc_monthly.groupby('cost_center')['cc_monthly_actual_spend_inr']
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    cc_monthly['rolling_std_3m_spend_inr'] = (
        cc_monthly.groupby('cost_center')['cc_monthly_actual_spend_inr']
        .transform(lambda s: s.rolling(window=3, min_periods=1).std().fillna(0.0))
    )
    cc_monthly['rolling_max_3m_spend_inr'] = (
        cc_monthly.groupby('cost_center')['cc_monthly_actual_spend_inr']
        .transform(lambda s: s.rolling(window=3, min_periods=1).max())
    )

    # Exponential Smoothing EWMA (alpha = 0.40)
    cc_monthly['ewma_spend_inr'] = (
        cc_monthly.groupby('cost_center')['cc_monthly_actual_spend_inr']
        .transform(lambda s: s.ewm(alpha=0.40, adjust=False).mean())
    )

    # Month-over-Month Growth Rate (%)
    safe_lag1 = np.maximum(1.0, cc_monthly['lag_1_month_spend_inr'].abs())
    cc_monthly['cc_mom_spend_growth_pct'] = np.clip(
        ((cc_monthly['cc_monthly_actual_spend_inr'] - cc_monthly['lag_1_month_spend_inr']) / safe_lag1) * 100.0,
        -95.0, 500.0
    )

    # Seasonality Index (Monthly Spend / Rolling Mean Spend)
    safe_roll = np.maximum(1.0, cc_monthly['rolling_mean_3m_spend_inr'].abs())
    cc_monthly['cc_seasonality_index'] = np.clip(
        cc_monthly['cc_monthly_actual_spend_inr'] / safe_roll, 0.10, 5.00
    )

    # Time-Series Spend Trend Slope (%)
    cc_monthly['cc_spend_trend_slope'] = np.clip(
        ((cc_monthly['ewma_spend_inr'] - cc_monthly['rolling_mean_3m_spend_inr']) / safe_roll) * 100.0,
        -80.0, 250.0
    )

    # Merge monthly time-series metrics back to line items
    df = df.merge(
        cc_monthly[[
            'cost_center', 'year_month',
            'cc_monthly_actual_spend_inr', 'cc_monthly_tx_count',
            'cc_monthly_mean_tx_inr', 'cc_monthly_max_tx_inr',
            'lag_1_month_spend_inr', 'lag_2_month_spend_inr', 'lag_3_month_spend_inr',
            'rolling_mean_3m_spend_inr', 'rolling_std_3m_spend_inr', 'rolling_max_3m_spend_inr',
            'ewma_spend_inr', 'cc_mom_spend_growth_pct', 'cc_seasonality_index', 'cc_spend_trend_slope'
        ]],
        on=['cost_center', 'year_month'],
        how='left'
    )

    # 6. Cost Center & Department Historical Baselines
    cc_hist_mean = df.groupby('cost_center')['actual_spend_inr'].transform('mean')
    cc_hist_std = df.groupby('cost_center')['actual_spend_inr'].transform('std').fillna(0.0)
    dept_hist_mean = df.groupby('department_spend_category')['actual_spend_inr'].transform('mean')

    df['cc_hist_mean_spend_inr'] = cc_hist_mean
    df['cc_hist_std_spend_inr'] = cc_hist_std
    df['dept_hist_mean_spend_inr'] = dept_hist_mean

    # Baseline Expected Spend per posting (combines CC mean, Dept mean, and Monthly Mean Tx)
    df['expected_spend_baseline_inr'] = np.round(
        0.50 * df['cc_monthly_mean_tx_inr'] + 0.35 * df['cc_hist_mean_spend_inr'] + 0.15 * df['dept_hist_mean_spend_inr'],
        2
    )

    # Statistical Z-score & Volatility Coefficient
    safe_cc_std = np.maximum(1.0, df['cc_hist_std_spend_inr'])
    df['cc_spend_zscore'] = np.clip((df['actual_spend_inr'] - df['cc_hist_mean_spend_inr']) / safe_cc_std, -3.5, 8.0)
    df['cc_spend_volatility_cv'] = np.clip(df['cc_hist_std_spend_inr'] / np.maximum(1.0, df['cc_hist_mean_spend_inr']), 0.0, 5.0)

    # Concentration and PO coverage features
    df['tx_share_of_monthly_spend'] = np.clip(
        df['actual_spend_inr'] / np.maximum(1.0, df['cc_monthly_actual_spend_inr']), 0.0, 1.0
    )
    df['po_commitment_to_spend_ratio'] = np.clip(
        df['cc_po_commitment_value_inr'] / np.maximum(1.0, df['cc_monthly_actual_spend_inr'] * 3.0), 0.0, 10.0
    )

    # Log transformations of monetary magnitudes
    df['log_actual_spend_inr'] = np.log1p(df['spend_abs_inr'])
    df['log_lag_1_month_spend_inr'] = np.log1p(np.maximum(0.0, df['lag_1_month_spend_inr']))
    df['log_lag_2_month_spend_inr'] = np.log1p(np.maximum(0.0, df['lag_2_month_spend_inr']))
    df['log_lag_3_month_spend_inr'] = np.log1p(np.maximum(0.0, df['lag_3_month_spend_inr']))
    df['log_rolling_mean_3m_spend_inr'] = np.log1p(np.maximum(0.0, df['rolling_mean_3m_spend_inr']))
    df['log_ewma_spend_inr'] = np.log1p(np.maximum(0.0, df['ewma_spend_inr']))
    df['log_po_commitment_inr'] = np.log1p(df['cc_po_commitment_value_inr'])

    df.drop(columns=['posting_date_dt'], inplace=True, errors='ignore')
    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extracts 32 domain-engineered numerical features for Time-Series Cost Center Spend Forecasting.
    """
    feature_cols = [
        'log_lag_1_month_spend_inr',
        'log_lag_2_month_spend_inr',
        'log_lag_3_month_spend_inr',
        'log_rolling_mean_3m_spend_inr',
        'log_ewma_spend_inr',
        'cc_mom_spend_growth_pct',
        'cc_seasonality_index',
        'cc_spend_trend_slope',
        'rolling_std_3m_spend_inr',
        'rolling_max_3m_spend_inr',
        'cc_monthly_tx_count',
        'cc_monthly_mean_tx_inr',
        'cc_monthly_max_tx_inr',
        'cc_hist_mean_spend_inr',
        'cc_hist_std_spend_inr',
        'dept_hist_mean_spend_inr',
        'expected_spend_baseline_inr',
        'cc_spend_zscore',
        'cc_spend_volatility_cv',
        'tx_share_of_monthly_spend',
        'posting_month',
        'posting_quarter',
        'posting_day_of_month',
        'posting_day_of_week',
        'is_weekend_posting',
        'is_month_end_window',
        'posting_lag_days',
        'month_sin',
        'month_cos',
        'log_po_commitment_inr',
        'po_commitment_to_spend_ratio',
        'cc_pm_order_count'
    ]

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['actual_spend_inr'].values
    return X, y, feature_cols


def prepare_time_series_train_test(df: pd.DataFrame, train_year: int = 2025, test_year: int = 2026):
    """
    Partitions the dataset chronologically by fiscal year:
    - Train Set: train_year (default 2025, 1 year of historical spend data)
    - Test / Forecast Set: test_year (default 2026, target forecast horizon)
    Applies RobustScaler fitted strictly on training data.
    """
    X, y, feature_cols = get_feature_matrix(df)

    is_train = (df['posting_year'] == train_year).values
    is_test = (df['posting_year'] == test_year).values

    # Fallback to 70/30 split if year filtering yields insufficient counts
    if is_train.sum() == 0 or is_test.sum() == 0:
        split_idx = int(len(df) * 0.70)
        is_train = np.zeros(len(df), dtype=bool)
        is_train[:split_idx] = True
        is_test = ~is_train

    X_train = X[is_train].copy()
    y_train = y[is_train].copy()
    X_test = X[is_test].copy()
    y_test = y[is_test].copy()

    idx_train = df.index[is_train]
    idx_test = df.index[is_test]

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


# Alias for backward compatibility
prepare_train_test_data = prepare_time_series_train_test
