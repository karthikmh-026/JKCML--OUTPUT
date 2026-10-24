"""
Inference and Reporting Module - Use Case 6: Slow Moving Inventory Prediction
Runs end-to-end scoring using the selected #1 Champion Model,
generates Probability of Obsolescence P(Slow Moving), Inventory Tiers,
and exports comprehensive executive Excel reports with Feature Importance,
Model Comparison, Obsolescence Watchlist, and Ignored Columns.
Ensures rigorous missing-value handling: text/metadata missing values are left blank ("").
"""
import os
import joblib
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


FEATURE_DESCRIPTIONS = {
    'days_since_last_issue': '(Days elapsed since last outward consumption / delivery goods movement [BWART 601, 261, 201])',
    'days_since_last_movement': '(Days elapsed since any goods movement in material document header [BUDAT_MKPF])',
    'days_since_creation': '(Material Master relationship tenure in days from creation date [MARA.ERSDA])',
    'is_inactive_90d': '(Binary indicator flagged when material has zero issue movements for >= 90 days)',
    'is_inactive_180d': '(Binary indicator flagged when material has zero issue movements for >= 180 days)',
    'is_zero_issue_window': '(Binary indicator for materials with zero consumption transactions across entire observation window)',
    'daily_consumption_rate': '(Average daily consumption quantity across the observation window)',
    'days_of_supply_forward': '(Forward-looking stock coverage in days = Current Stock / Daily Consumption Rate)',
    'inventory_turnover_ratio': '(Turnover velocity ratio = Total Issue Quantity / Current Stock Quantity)',
    'annualized_turnover_rate': '(Annualized inventory turnover velocity scaled to 365-day basis)',
    'stock_value_log': '(Log-transformed total financial valuation of current inventory in INR [SALK3])',
    'current_stock_qty_log': '(Log-transformed physical stock quantity on hand [LBKUM])',
    'unit_valuation_price': '(Effective unit price per base unit of measure = Stock Value / Stock Quantity)',
    'price_variance_ratio': '(Discrepancy ratio between moving average price [VERPR] and standard price [STPRS])',
    'high_value_stagnant_flag': '(Flag for stagnant inventory locking up >= 500,000 INR with >= 90 days inactivity)',
    'receipt_to_issue_ratio': '(Procurement imbalance ratio measuring inward receipts relative to outward issues)',
    'net_accumulation_qty': '(Net physical stock accumulation = Inward Receipts minus Outward Issues)',
    'stock_drift_previous_period': '(Percentage drift in stock quantity relative to previous period valuation [VMKUM])',
    'stock_drift_previous_year': '(Percentage drift in stock quantity relative to previous fiscal year valuation [VJKUM])',
    'stock_to_safety_ratio': '(Multiple of current stock relative to plant safety stock buffer [MARC.EISBE])',
    'excess_safety_stock_qty': '(Physical stock quantity exceeding established safety buffer)',
    'excess_safety_stock_val': '(Financial capital locked in stock exceeding established safety buffer)',
    'reorder_point_gap': '(Difference between current stock level and plant reorder point [MARC.MINBE])',
    'issue_transaction_density': '(Proportion of total goods movements that represent consumption / issue activities)',
    'inactivity_rate_window': '(Proportion of calendar year represented by inactive days)',
    'obsolescence_risk_index': '(Composite multi-criteria index weighting inactivity, days of supply, and zero consumption)',
    'capital_lockup_strain': '(Financial severity metric scaling capital valuation by obsolescence dormancy)',
    'abc_numeric': '(Numeric rank of SAP ABC classification: A=3, B=2, C=1, Unclassified=0)',
    'planned_deliv_days': '(Vendor/production planned delivery lead time in calendar days [MARC.PLIFZ])',
    'net_weight_log': '(Log-transformed material physical net weight in kg [MARA.NTGEW])',
    'mrp_type_encoded': '(Frequency-weighted encoding of SAP MRP Type [MARC.DISMM])',
    'material_type_encoded': '(Frequency-weighted encoding of SAP Material Type [MARA.MTART])'
}

IGNORED_COLUMNS_INFO = [
    ("Material Number", "MATNR", "Unique alphanumeric SAP Material Master primary key. Carries no intrinsic mathematical signal on physical demand or obsolescence."),
    ("Material Description Text", "MAKTX", "Free-text description string retained strictly for supply chain planner identification and human reporting."),
    ("Plant / Valuation Area", "BWKEY, WERKS", "Organizational entity codes retained for plant filtering, location reporting, and multi-facility governance."),
    ("Plant Name Text", "NAME1", "Descriptive textual name of manufacturing plant or regional distribution center."),
    ("Raw Calendar Timestamps", "ERSDA, LAEDA, BUDAT_MKPF", "Calendar dates converted into relative temporal features (days since creation, days since last issue, days of supply) for time-invariance."),
    ("Document Line Item Counters", "ZEILE, MBLNR, MJAHR", "Internal transactional tracking IDs with zero predictive value for demand estimation."),
    ("Unit of Measure Strings", "MEINS, WAERS", "Descriptive unit codes (TO, KG, EA, INR) retained for unit reporting.")
]


def score_with_single_model(model, df_featured: pd.DataFrame, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the fitted #1 Champion model to score all material records across HANA_P2 and HANA_P3.
    Returns enriched DataFrame with predictions, probabilities, and actionable recommendations.
    Standardizes missing text as blank strings ("").
    """
    X = df_featured[feature_cols].values
    X_scaled = scaler.transform(X)

    tiers, tier_names = model.predict_clusters(X_scaled)
    prob_scores, raw_scores = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)
    categories = model.assign_risk_category(prob_scores)

    df_out = df_featured.copy()
    df_out['predicted_inventory_tier'] = tiers
    df_out['inventory_tier_name'] = tier_names
    df_out['probability_of_obsolescence'] = np.round(prob_scores, 4)
    df_out['slow_moving_risk_score'] = np.round(raw_scores, 4)
    df_out['predicted_slow_moving_exception'] = preds
    df_out['inventory_risk_category'] = categories

    def assign_action(row):
        cat = row['inventory_risk_category']
        days = row['days_since_last_issue']
        val = row['current_stock_val']

        if "Critical" in cat or days >= 180.0:
            if val > 500000.0:
                return "CRITICAL ACTION: High-value dead stock. Halt POs, initiate write-down review, execute bulk salvage/liquidation"
            else:
                return "CRITICAL ACTION: Dormant stock. Halt replenishment, scrap or salvage low-value obsolete units"
        elif "High" in cat or days >= 90.0:
            return "HIGH PRIORITY: Slow-moving inventory. Freeze reorder points, reallocate to active plants, discount/bundle"
        elif "Moderate" in cat:
            return "MODERATE WATCH: Slower turnover. Review safety stock parameters, monitor consumption trends for 60 days"
        else:
            return "NORMAL / ACTIVE: Fast-moving healthy inventory. Maintain standard replenishment and lead-time buffers"

    df_out['prescriptive_action'] = df_out.apply(assign_action, axis=1)
    return df_out


def export_enriched_excel_report(
    scored_df: pd.DataFrame,
    output_path: str,
    fi_df: pd.DataFrame = None,
    comp_df: pd.DataFrame = None
):
    """
    Exports a styled 3-sheet executive Excel report for the entire dataset:
    1. Enriched_Inventory_Predictions
    2. Model_Performance_Ranking
    3. Feature_Importance_And_Metadata
    Optimized for high-throughput export (>200,000 rows).
    Missing values are cleanly exported as blanks ("").
    """
    print(f"\n[REPORT] Generating styled executive Excel workbook: {output_path} (Records: {len(scored_df):,})...")

    export_cols = [
        'source_schema', 'matnr', 'plant', 'plant_name', 'material_name', 'material_type', 'material_group', 'base_uom',
        'current_stock_qty', 'current_stock_val', 'moving_avg_price', 'standard_price',
        'days_since_last_issue', 'days_since_last_movement', 'daily_consumption_rate', 'days_of_supply_forward',
        'inventory_turnover_ratio', 'issue_txn_cnt', 'total_issue_qty', 'total_receipt_qty',
        'safety_stock', 'excess_safety_stock_val',
        'predicted_inventory_tier', 'inventory_tier_name',
        'probability_of_obsolescence', 'slow_moving_risk_score',
        'predicted_slow_moving_exception', 'inventory_risk_category',
        'prescriptive_action'
    ]
    export_cols = [c for c in export_cols if c in scored_df.columns]

    df_export = scored_df[export_cols].copy()

    # Clean missing values: replace NaN/None with blank string ""
    for col in df_export.select_dtypes(include=['object', 'string']).columns:
        df_export[col] = df_export[col].fillna("").astype(str).str.strip()
        df_export[col] = df_export[col].replace({'None': '', 'nan': '', 'NaN': '', 'null': '', 'NULL': ''})

    for col in df_export.select_dtypes(include=['float', 'int', 'number']).columns:
        df_export[col] = df_export[col].fillna(0.0)

    # Use pandas ExcelWriter with openpyxl engine for fast 200k+ row export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Enriched Inventory Predictions
        df_export.to_excel(writer, sheet_name="Enriched_Inventory_Predictions", index=False)

        # Sheet 2: Model Performance Ranking
        if comp_df is not None:
            comp_df.to_excel(writer, sheet_name="Model_Performance_Ranking", index=False)

        # Sheet 3: Feature Importance & Metadata
        if fi_df is not None:
            fi_df_export = fi_df.copy()
            fi_df_export['Domain Supply Chain Interpretation'] = fi_df_export['Feature'].map(
                lambda f: FEATURE_DESCRIPTIONS.get(f, "Engineered inventory supply chain signal")
            )
            fi_df_export.to_excel(writer, sheet_name="Feature_Importance_And_Metadata", index=False)

    # Apply executive styling to headers
    wb = openpyxl.load_workbook(output_path)
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    sub_fill = PatternFill(start_color="244062", end_color="244062", fill_type="solid")

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        fill = header_fill if sheet_name == "Enriched_Inventory_Predictions" else sub_fill
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            col_letter = get_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = 22

    # Append Ignored Columns documentation to Sheet 3
    if "Feature_Importance_And_Metadata" in wb.sheetnames:
        ws3 = wb["Feature_Importance_And_Metadata"]
        start_row = ws3.max_row + 4
        ws3.cell(row=start_row, column=1, value="SAP Raw Technical Columns Ignored / Transformed").font = Font(name="Calibri", size=13, bold=True, color="1F497D")
        ws3.cell(row=start_row + 1, column=1, value="Attribute Category").font = header_font
        ws3.cell(row=start_row + 1, column=1).fill = sub_fill
        ws3.cell(row=start_row + 1, column=2, value="SAP Columns").font = header_font
        ws3.cell(row=start_row + 1, column=2).fill = sub_fill
        ws3.cell(row=start_row + 1, column=3, value="Exclusion / Transformation Rationale").font = header_font
        ws3.cell(row=start_row + 1, column=3).fill = sub_fill

        for idx, (cat, cols, rat) in enumerate(IGNORED_COLUMNS_INFO, start=start_row + 2):
            ws3.cell(row=idx, column=1, value=cat)
            ws3.cell(row=idx, column=2, value=cols)
            ws3.cell(row=idx, column=3, value=rat)

    wb.save(output_path)
    print(f"[REPORT] Successfully generated Excel workbook at: {output_path}")


def save_single_model_pipeline(
    champion_model,
    scaler,
    feature_cols: list,
    output_dir: str = "models"
):
    """Persists champion model, scaler, and full pipeline bundle to models/."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "champion_slow_moving_model.joblib")
    scaler_path = os.path.join(output_dir, "slow_moving_scaler.joblib")
    bundle_path = os.path.join(output_dir, "slow_moving_model_bundle.joblib")

    joblib.dump(champion_model, model_path)
    joblib.dump(scaler, scaler_path)

    bundle = {
        'model': champion_model,
        'model_name': champion_model.model_name,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'n_clusters': champion_model.n_clusters,
        'threshold': champion_model.threshold,
        'rank_map': champion_model.rank_map,
        'ordered_centroids': champion_model.ordered_centroids
    }
    joblib.dump(bundle, bundle_path)

    print(f"[ARTIFACTS] Successfully persisted Champion Model to {model_path}")
    print(f"[ARTIFACTS] Successfully persisted Scaler to {scaler_path}")
    print(f"[ARTIFACTS] Successfully persisted Full Bundle to {bundle_path}")
