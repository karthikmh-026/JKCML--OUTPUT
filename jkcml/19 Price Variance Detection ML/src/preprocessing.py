"""
Preprocessing & Feature Engineering Module
Transforms raw SAP tables into 32 structured domain features for Price Variance Anomaly Detection.
Ensures rigorous missing-value preservation so missing data exports as true blanks in Excel.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw procurement data and creates 32 domain-specific features for price variance detection.
    Missing master-data and contractual fields are preserved as np.nan so they export as genuine blanks.
    """
    df = df.copy()

    # 1. Clean string identifiers and treat literal 'None', 'nan', '' as true np.nan
    string_cols = ['vendor_region', 'vendor_city', 'plant_name', 'plant_city', 'material_desc', 'material_type', 'unit_of_measure']
    for c in string_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({'None': np.nan, 'nan': np.nan, 'NaN': np.nan, '': np.nan, '<NA>': np.nan})

    df['material_id'] = df['material_id'].astype(str).str.strip()
    df['material_group'] = df['material_group'].astype(str).str.strip()
    df['vendor_id'] = df['vendor_id'].astype(str).str.strip()
    df['plant'] = df['plant'].astype(str).str.strip()

    # 2. Normalize PO unit price by price unit (PEINH)
    df['po_price_unit'] = pd.to_numeric(df['po_price_unit'], errors='coerce').fillna(1.0).replace(0, 1.0)
    df['po_unit_price'] = pd.to_numeric(df['po_unit_price'], errors='coerce').fillna(0.0)
    df['effective_po_price'] = df['po_unit_price'] / df['po_price_unit']

    # 3. Contract / Info Record Price Handling
    # An active contract exists ONLY if info_rec_price is present and > 0, with a valid info_record_id
    df['info_rec_price'] = pd.to_numeric(df['info_rec_price'], errors='coerce')
    df['info_rec_price_unit'] = pd.to_numeric(df['info_rec_price_unit'], errors='coerce')

    has_contract_mask = (
        df['info_rec_price'].notna() &
        (df['info_rec_price'] > 0) &
        df['info_record_id'].notna() &
        (df['info_record_id'].astype(str).str.strip() != '') &
        (df['info_record_id'].astype(str).str.lower() != 'none') &
        (df['info_record_id'].astype(str).str.lower() != 'nan')
    )
    df['has_info_contract'] = has_contract_mask.astype(int)

    # For valid contract rows: compute effective contract price
    # For non-contract rows: strictly keep as np.nan so they export as true BLANKS in Excel!
    unit_clean = df['info_rec_price_unit'].fillna(1.0).replace(0, 1.0)
    df['effective_info_price'] = np.where(has_contract_mask, df['info_rec_price'] / unit_clean, np.nan)
    df['info_rec_price'] = np.where(has_contract_mask, df['info_rec_price'], np.nan)
    df['info_rec_price_unit'] = np.where(has_contract_mask, df['info_rec_price_unit'].fillna(1.0), np.nan)
    df['info_record_id'] = np.where(has_contract_mask, df['info_record_id'].astype(str).str.strip(), np.nan)

    # 4. Compute benchmark reference prices
    mat_medians = df.groupby('material_id')['effective_po_price'].transform('median')
    mat_means = df.groupby('material_id')['effective_po_price'].transform('mean')
    mat_stds = df.groupby('material_id')['effective_po_price'].transform('std').fillna(1.0).replace(0, 1.0)

    grp_medians = df.groupby('material_group')['effective_po_price'].transform('median')
    grp_means = df.groupby('material_group')['effective_po_price'].transform('mean')
    grp_stds = df.groupby('material_group')['effective_po_price'].transform('std').fillna(1.0).replace(0, 1.0)

    plant_mat_medians = df.groupby(['plant', 'material_id'])['effective_po_price'].transform('median')

    # Reference price: Prioritize Contract Info Price -> Material Median -> Group Median -> Effective PO Price
    effective_info_num = np.nan_to_num(df['effective_info_price'].values, nan=0.0)
    df['reference_price'] = np.where(
        df['has_info_contract'] == 1,
        effective_info_num,
        np.where(~mat_medians.isna(), mat_medians, grp_medians)
    )
    df['reference_price'] = df['reference_price'].fillna(df['effective_po_price']).replace(0, 0.01)

    # 5. Price & Variance Ratio Metrics
    df['price_variance_abs'] = df['effective_po_price'] - df['reference_price']
    df['price_variance_pct'] = ((df['effective_po_price'] - df['reference_price']) / df['reference_price']) * 100.0
    df['price_ratio'] = df['effective_po_price'] / df['reference_price']
    df['log_price_ratio'] = np.log1p(np.maximum(0, df['price_ratio']))

    # 6. Contract Compliance Metrics
    contract_gap = np.where(df['has_info_contract'] == 1, df['effective_po_price'] - effective_info_num, 0.0)
    df['contract_price_diff'] = contract_gap
    df['contract_variance_pct'] = np.where(
        df['has_info_contract'] == 1,
        ((df['effective_po_price'] - effective_info_num) / np.maximum(effective_info_num, 0.01)) * 100.0,
        0.0
    )
    df['has_contract_leakage'] = ((df['has_info_contract'] == 1) & (df['effective_po_price'] > effective_info_num)).astype(int)

    # 7. Statistical Distribution & Z-scores
    df['group_price_zscore'] = ((df['effective_po_price'] - grp_means) / grp_stds).clip(-10, 10).fillna(0)
    df['mat_price_zscore'] = ((df['effective_po_price'] - mat_means) / mat_stds).clip(-10, 10).fillna(0)
    df['price_to_group_median_ratio'] = (df['effective_po_price'] / grp_medians.replace(0, 0.01)).clip(0, 50).fillna(1.0)
    df['price_to_mat_median_ratio'] = (df['effective_po_price'] / mat_medians.replace(0, 0.01)).clip(0, 50).fillna(1.0)

    # 8. Vendor Behavioral Metrics
    df['vendor_price_diff'] = df['effective_po_price'] - mat_medians.fillna(df['effective_po_price'])
    vendor_grp_means = df.groupby(['vendor_id', 'material_group'])['effective_po_price'].transform('mean')
    df['vendor_to_group_ratio'] = (vendor_grp_means / grp_medians.replace(0, 0.01)).clip(0, 50).fillna(1.0)
    vendor_cnt = df.groupby('vendor_id')['po_number'].transform('count')
    df['vendor_order_frequency'] = np.log1p(vendor_cnt)
    vendor_mat_cnt = df.groupby(['vendor_id', 'material_id'])['po_number'].transform('count')
    df['vendor_item_share'] = (vendor_mat_cnt / vendor_cnt.replace(0, 1)).clip(0, 1.0)

    # 9. Plant & Facility Metrics
    df['plant_mat_price_diff'] = df['effective_po_price'] - plant_mat_medians.fillna(mat_medians).fillna(df['effective_po_price'])
    df['price_to_plant_median_ratio'] = (df['effective_po_price'] / plant_mat_medians.replace(0, 0.01)).clip(0, 50).fillna(1.0)

    # 10. Volume, Scale & Spend Intensity
    df['order_quantity'] = pd.to_numeric(df['order_quantity'], errors='coerce').fillna(0.0)
    df['log_order_qty'] = np.log1p(np.maximum(0, df['order_quantity']))
    df['po_net_value'] = pd.to_numeric(df['po_net_value'], errors='coerce').fillna(df['effective_po_price'] * df['order_quantity'])
    df['log_net_value'] = np.log1p(np.maximum(0, df['po_net_value']))
    df['spend_intensity'] = (df['po_net_value'] / df['order_quantity'].replace(0, 1)).clip(0, 1e7)

    spend_95 = df['po_net_value'].quantile(0.95)
    df['is_high_spend_anomaly'] = (df['po_net_value'] > spend_95).astype(int)

    # 11. Domestic vs Foreign Procurement
    df['is_domestic_vendor'] = (df['vendor_country'].fillna('IN').astype(str).str.upper() == 'IN').astype(int)

    # 12. Temporal & Cyclical Indicators
    po_dt = pd.to_datetime(df['po_date'], errors='coerce')
    df['po_month'] = po_dt.dt.month.fillna(6).astype(int)
    df['po_quarter'] = po_dt.dt.quarter.fillna(2).astype(int)
    df['is_month_end'] = (po_dt.dt.day >= 25).astype(int)

    # 13. Business Ground Truth / Exception Target (Y)
    condition_contract_spike = (df['has_info_contract'] == 1) & (df['price_variance_pct'] > 15.0)
    condition_market_spike = (df['price_variance_pct'] > 25.0)
    condition_zscore_spike = (df['group_price_zscore'] > 3.0)

    df['is_variance_exception'] = (
        condition_contract_spike | condition_market_spike | condition_zscore_spike
    ).astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extracts 32 numerical domain features for Price Variance Detection.
    Fills NaNs with 0 for model calculation while keeping df pristine.
    """
    feature_cols = [
        'effective_po_price',
        'has_info_contract',
        'price_variance_pct',
        'log_price_ratio',
        'price_ratio',
        'price_variance_abs',
        'reference_price',
        'contract_price_diff',
        'contract_variance_pct',
        'has_contract_leakage',
        'group_price_zscore',
        'mat_price_zscore',
        'price_to_group_median_ratio',
        'price_to_mat_median_ratio',
        'vendor_price_diff',
        'vendor_to_group_ratio',
        'plant_mat_price_diff',
        'price_to_plant_median_ratio',
        'log_net_value',
        'po_net_value',
        'log_order_qty',
        'order_quantity',
        'spend_intensity',
        'po_price_unit',
        'effective_info_price',
        'vendor_order_frequency',
        'vendor_item_share',
        'is_high_spend_anomaly',
        'is_domestic_vendor',
        'po_month',
        'po_quarter',
        'is_month_end'
    ]

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['is_variance_exception'].values
    return X, y, feature_cols


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Extracts features, performs train/test split, and applies RobustScaler.
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
