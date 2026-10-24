"""
Preprocessing & Feature Engineering Module - Use Case 21: Material Master Data Quality Scoring
Transforms raw SAP Material Master tables (MARA, MAKT, MARC, MBEW, MARM, EINA, T134T, T001W, T001)
across HANA_P2 and HANA_P3 into 32 structured domain features for Unsupervised Data Quality Scoring.
Includes rigorous missing value handling (putting clean blanks where data is absent).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


NON_VALUATED_TYPES = {'ZNVM', 'ZFOC', 'ZMES', 'ZCFM'}
PHYSICAL_WEIGHT_MANDATORY_TYPES = {'ZFIN', 'ZTAC', 'ZTPC', 'ZTRA', 'ZPPM', 'ZTWP', 'ZCHM'}
COMMERCIAL_HIERARCHY_TYPES = {'ZFIN', 'ZTAC', 'ZTPC', 'ZTRA', 'ZTWP', 'ZCHM'}


def clean_missing_str(series: pd.Series) -> pd.Series:
    """
    Cleans string columns: converts None, NaN, 'nan', 'none', 'null', '<NA>'
    to clean blank string '', preserving actual data when present.
    """
    s = series.fillna('').astype(str).str.strip()
    mask_empty = s.str.lower().isin({'nan', 'none', 'null', 'nat', '<na>', 'undefined', ''})
    return s.mask(mask_empty, '')


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw SAP Material Master data across Basic (MARA/MAKT), Plant/MRP/Purchasing (MARC),
    and Accounting/Valuation (MBEW) views with thorough missing-value handling,
    and engineers 32 domain-specific features for detecting incomplete or incorrect material records.
    """
    df = df.copy()

    # 1. Clean string attributes across MARA, MAKT, MARC, MBEW
    str_cols = [
        'source_schema', 'company_code', 'company_name', 'company_city',
        'company_country', 'company_currency', 'material_id', 'material_desc',
        'material_type', 'material_type_desc', 'industry_sector', 'material_group',
        'old_material_no', 'base_uom', 'order_uom', 'division', 'prod_hierarchy',
        'trans_group', 'batch_mgmt_flag', 'cross_plant_status', 'complete_maint_status',
        'mara_maint_status', 'mara_deletion_flag', 'weight_unit', 'volume_unit',
        'created_by', 'changed_by', 'plant', 'plant_name', 'plant_city',
        'plant_region', 'plant_country', 'marc_maint_status', 'marc_deletion_flag',
        'plant_mat_status', 'abc_indicator', 'purchasing_group', 'mrp_type',
        'mrp_controller', 'procurement_type', 'profit_center', 'loading_group',
        'avail_check_group', 'valuation_class', 'price_control', 'mbew_deletion_flag'
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = clean_missing_str(df[col])
        else:
            df[col] = ''

    # Default company and plant names if master record has company code 1000
    df['company_code'] = df['company_code'].replace('', '1000')
    df['company_name'] = df['company_name'].replace('', 'J.K. CEMENT LIMITED')
    df['company_city'] = df['company_city'].replace('', 'Kanpur')
    df['company_country'] = df['company_country'].replace('', 'IN')
    df['company_currency'] = df['company_currency'].replace('', 'INR')

    # 2. Clean numeric attributes across MARA, MARC, MBEW
    num_cols = [
        'gross_weight', 'net_weight', 'volume', 'planned_deliv_days',
        'gr_processing_days', 'reorder_point', 'safety_stock', 'min_lot_size',
        'max_lot_size', 'costing_lot_size', 'moving_avg_price', 'standard_price',
        'price_unit', 'total_valuated_stock', 'total_stock_value',
        'prev_moving_price', 'prev_standard_price', 'alt_uom_count', 'info_record_count'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    df['price_unit'] = df['price_unit'].replace(0, 1.0).clip(lower=1.0)

    # 3. Valuation & Accounting Integrity Features (MBEW vs MARA)
    is_valuated_type = (~df['material_type'].isin(NON_VALUATED_TYPES)).astype(int)

    df['is_missing_valuation_view'] = (
        (is_valuated_type == 1) &
        ((df['valuation_class'] == '') | (df['price_control'] == ''))
    ).astype(int)

    df['is_zero_valuation_price'] = (
        (is_valuated_type == 1) &
        (df['moving_avg_price'] <= 0.0) &
        (df['standard_price'] <= 0.0)
    ).astype(int)

    df['is_price_control_mismatch'] = (
        (is_valuated_type == 1) & (
            ((df['price_control'] == 'S') & (df['standard_price'] <= 0.0)) |
            ((df['price_control'] == 'V') & (df['moving_avg_price'] <= 0.0))
        )
    ).astype(int)

    df['is_negative_stock_or_value'] = (
        (df['total_valuated_stock'] < 0.0) | (df['total_stock_value'] < 0.0)
    ).astype(int)

    df['is_stock_value_inconsistent'] = (
        ((df['total_valuated_stock'] > 0.0) & (df['total_stock_value'] <= 0.0)) |
        ((df['total_valuated_stock'] == 0.0) & (df['total_stock_value'].abs() > 1.0))
    ).astype(int)

    # Effective Unit Price & Book vs Implied Stock Valuation Discrepancy
    raw_eff_price = np.where(
        df['price_control'] == 'S',
        np.where(df['standard_price'] > 0, df['standard_price'], df['moving_avg_price']),
        np.where(df['moving_avg_price'] > 0, df['moving_avg_price'], df['standard_price'])
    ) / df['price_unit']
    df['effective_unit_price'] = np.clip(raw_eff_price, 0.0, 1e8)
    df['log_effective_price'] = np.log1p(df['effective_unit_price'])

    implied_unit_val = np.where(
        df['total_valuated_stock'] > 0,
        df['total_stock_value'] / np.maximum(df['total_valuated_stock'], 1e-6),
        df['effective_unit_price']
    )
    df['implied_vs_book_price_diff_pct'] = np.where(
        (df['total_valuated_stock'] > 0) & (df['effective_unit_price'] > 0),
        np.abs(implied_unit_val - df['effective_unit_price']) / df['effective_unit_price'] * 100.0,
        0.0
    ).clip(0.0, 500.0)

    # Material Group Peer Price Z-Score
    mg_mean = df.groupby('material_group')['log_effective_price'].transform('mean')
    mg_std = df.groupby('material_group')['log_effective_price'].transform('std').fillna(1.0).replace(0, 1.0)
    df['mat_group_price_zscore'] = ((df['log_effective_price'] - mg_mean) / mg_std).clip(-10.0, 10.0).fillna(0.0)

    df['log_valuated_stock'] = np.log1p(df['total_valuated_stock'].abs())
    df['log_stock_value'] = np.log1p(df['total_stock_value'].abs())

    # 4. Plant, MRP & Purchasing Governance Features (MARC)
    is_purch_type = (~df['material_type'].isin({'ZFIN', 'ZTAC', 'ZSCR'})).astype(int)
    df['is_missing_purchasing_group'] = (
        (is_purch_type == 1) & (df['purchasing_group'] == '')
    ).astype(int)

    df['is_mrp_governance_incomplete'] = (
        (df['mrp_type'] == '') |
        ((df['mrp_type'] != 'ND') & ((df['mrp_controller'] == '') | (df['procurement_type'] == ''))) |
        ((df['mrp_type'] == 'ND') & (df['mrp_controller'] == '') & (df['procurement_type'] == '') & (df['material_type'].isin({'ZRAW', 'ZSEM', 'ZPAC', 'ZPPM'})))
    ).astype(int)

    df['is_missing_profit_center'] = (
        (df['profit_center'] == '') | (df['profit_center'] == '0000000000')
    ).astype(int)

    # 5. Physical Weight & Dimension Completeness / Consistency (MARA)
    is_weight_req = df['material_type'].isin(PHYSICAL_WEIGHT_MANDATORY_TYPES).astype(int)
    df['is_missing_weight_spec'] = (
        (is_weight_req == 1) &
        ((df['gross_weight'] <= 0.0) | (df['net_weight'] <= 0.0) | (df['weight_unit'] == ''))
    ).astype(int)

    df['is_weight_inconsistent'] = (
        ((df['gross_weight'] > 0.0) & (df['net_weight'] > df['gross_weight'])) |
        ((df['gross_weight'] > 0.0) & (df['net_weight'] <= 0.0)) |
        ((df['net_weight'] > 0.0) & (df['gross_weight'] <= 0.0))
    ).astype(int)

    # 6. Material Description Quality & Duplicate Standardization Checks (MAKT)
    norm_desc = (
        df['material_desc']
        .str.upper()
        .str.replace(r'[^A-Z0-9]', '', regex=True)
    )
    df['_norm_desc'] = np.where(norm_desc == '', 'EMPTY_DESC', norm_desc)

    dup_mat_cnt = df.groupby(['material_type', 'material_group', '_norm_desc'])['material_id'].transform('nunique')
    df['dup_description_count'] = dup_mat_cnt
    df['is_duplicate_description'] = (
        (dup_mat_cnt >= 4) & (df['_norm_desc'] != 'EMPTY_DESC')
    ).astype(int)

    df['desc_char_length'] = df['material_desc'].str.len()
    df['desc_token_count'] = df['material_desc'].str.split().str.len().fillna(0)
    df['is_cryptic_or_short_desc'] = (
        (df['desc_char_length'] < 7) |
        (df['desc_token_count'] <= 1) |
        (df['material_desc'].str.upper().isin({'TEST', 'DUMMY', 'NA', 'N/A', 'TBD', 'SAMPLE'}))
    ).astype(int)

    # 7. General Master Attribute Completeness & Maintenance View Synchronization (MARA / MARC)
    df['is_missing_old_mat_no'] = (df['old_material_no'] == '').astype(int)
    df['is_missing_division'] = (df['division'] == '').astype(int)
    df['is_missing_prod_hierarchy'] = (
        df['material_type'].isin(COMMERCIAL_HIERARCHY_TYPES) & (df['prod_hierarchy'] == '')
    ).astype(int)

    df['mara_views_maintained_cnt'] = df['mara_maint_status'].str.len().clip(lower=1)
    df['marc_views_maintained_cnt'] = df['marc_maint_status'].str.len().clip(lower=1)
    vpsta_len = df['complete_maint_status'].str.len().clip(lower=1)
    df['view_sync_gap'] = (vpsta_len - df['marc_views_maintained_cnt']).clip(lower=0)

    # Count of missing critical / governance fields across the 12 core master attributes
    df['missing_critical_fields_cnt'] = (
        df['is_missing_valuation_view'] +
        df['is_zero_valuation_price'] +
        df['is_missing_purchasing_group'] +
        df['is_mrp_governance_incomplete'] +
        df['is_missing_profit_center'] +
        df['is_missing_weight_spec'] +
        df['is_missing_old_mat_no'] +
        df['is_missing_division'] +
        df['is_missing_prod_hierarchy'] +
        (df['material_group'] == '').astype(int) +
        (df['base_uom'] == '').astype(int) +
        (df['loading_group'] == '').astype(int)
    )
    df['completeness_ratio'] = np.clip((12.0 - df['missing_critical_fields_cnt']) / 12.0, 0.0, 1.0)

    # 8. Temporal Maintenance Lag (ERSDA to LAEDA)
    created_dt = pd.to_datetime(df['created_date'], errors='coerce')
    changed_dt = pd.to_datetime(df['last_changed_date'], errors='coerce').fillna(created_dt)
    created_dt = created_dt.fillna(changed_dt)
    df['days_since_last_change'] = (changed_dt - created_dt).dt.days.fillna(0).clip(0, 3650)

    # 9. Composite Material Master Data Quality Defect Index (0 to 100 scale) & Health Score (100 - Defect)
    raw_defect = (
        34.0 * df['is_missing_valuation_view'] +
        26.0 * df['is_zero_valuation_price'] +
        24.0 * df['is_price_control_mismatch'] +
        38.0 * df['is_negative_stock_or_value'] +
        32.0 * df['is_stock_value_inconsistent'] +
        28.0 * df['is_missing_purchasing_group'] +
        22.0 * df['is_mrp_governance_incomplete'] +
        25.0 * df['is_missing_profit_center'] +
        22.0 * df['is_missing_weight_spec'] +
        28.0 * df['is_weight_inconsistent'] +
        14.0 * df['is_duplicate_description'] +
        14.0 * df['is_cryptic_or_short_desc'] +
        4.0 * df['is_missing_old_mat_no'] +
        3.0 * df['is_missing_division'] +
        6.0 * df['is_missing_prod_hierarchy'] +
        1.5 * np.minimum(6, df['view_sync_gap'])
    )
    df['mdq_defect_index'] = np.clip(raw_defect, 0.0, 100.0)
    df['mdq_health_score'] = np.round(100.0 - df['mdq_defect_index'], 2)

    # 10. Classify Human-Readable Material Quality Issue Type for Executive Reporting
    issue_conditions = [
        (df['is_negative_stock_or_value'] == 1) | (df['is_stock_value_inconsistent'] == 1),
        (df['is_missing_valuation_view'] == 1),
        (df['is_price_control_mismatch'] == 1) | (df['is_zero_valuation_price'] == 1),
        (df['is_weight_inconsistent'] == 1) | (df['is_missing_weight_spec'] == 1),
        (df['is_missing_purchasing_group'] == 1),
        (df['is_mrp_governance_incomplete'] == 1),
        (df['is_duplicate_description'] == 1) & (df['is_cryptic_or_short_desc'] == 1),
        (df['is_duplicate_description'] == 1),
        (df['is_cryptic_or_short_desc'] == 1)
    ]
    issue_choices = [
        'Stock / Valuation Balance Corruption (MBEW)',
        'Missing Valuation Class / Accounting View (BKLAS)',
        'Zero / Mismatched Valuation Price (STPRS/VERPR)',
        'Missing or Inconsistent Weight Spec (BRGEW/NTGEW)',
        'Missing Purchasing Group Governance (EKGRP)',
        'Incomplete MRP Controller / Type Setup (DISMM/DISPO)',
        'Duplicate & Cryptic Material Description (MAKTX)',
        'High-Frequency Duplicate Material Description',
        'Cryptic / Incomplete Material Description'
    ]
    df['quality_issue_type'] = np.select(issue_conditions, issue_choices, default='Compliant Material Record')

    # 11. Post-Hoc Audit Benchmark Ground-Truth Exception Flag (Y)
    # Strictly excluded from unsupervised training; used ONLY to evaluate and rank the 5 ML models
    cond_val_defect = (
        (df['is_missing_valuation_view'] == 1) |
        (df['is_zero_valuation_price'] == 1) |
        (df['is_price_control_mismatch'] == 1) |
        (df['is_negative_stock_or_value'] == 1) |
        (df['is_stock_value_inconsistent'] == 1)
    )
    cond_gov_defect = (
        (df['is_missing_purchasing_group'] == 1) |
        (df['is_mrp_governance_incomplete'] == 1) |
        (df['is_missing_profit_center'] == 1)
    )
    cond_phys_defect = (
        (df['is_missing_weight_spec'] == 1) |
        (df['is_weight_inconsistent'] == 1)
    )
    cond_desc_multi_defect = (
        (df['is_duplicate_description'] == 1) &
        (df['is_cryptic_or_short_desc'] == 1) &
        (df['missing_critical_fields_cnt'] >= 2)
    )

    df['is_quality_exception'] = (
        cond_val_defect | cond_gov_defect | cond_phys_defect | cond_desc_multi_defect
    ).astype(int)

    df.drop(columns=['_norm_desc'], inplace=True, errors='ignore')
    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extracts 32 numerical domain features for Unsupervised Material Master Data Quality Scoring.
    """
    feature_cols = [
        'mdq_defect_index',
        'mdq_health_score',
        'missing_critical_fields_cnt',
        'completeness_ratio',
        'is_missing_valuation_view',
        'is_zero_valuation_price',
        'is_price_control_mismatch',
        'is_negative_stock_or_value',
        'is_stock_value_inconsistent',
        'is_missing_purchasing_group',
        'is_mrp_governance_incomplete',
        'is_missing_profit_center',
        'is_missing_weight_spec',
        'is_weight_inconsistent',
        'is_duplicate_description',
        'dup_description_count',
        'is_cryptic_or_short_desc',
        'desc_char_length',
        'desc_token_count',
        'is_missing_old_mat_no',
        'is_missing_division',
        'is_missing_prod_hierarchy',
        'mara_views_maintained_cnt',
        'marc_views_maintained_cnt',
        'view_sync_gap',
        'effective_unit_price',
        'log_effective_price',
        'mat_group_price_zscore',
        'implied_vs_book_price_diff_pct',
        'log_valuated_stock',
        'log_stock_value',
        'days_since_last_change'
    ]

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['is_quality_exception'].values
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
