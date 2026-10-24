"""
Inference and Reporting Module - Use Case 7: Production Order Delay Prediction
Runs end-to-end scoring using the selected #1 Champion Model,
generates continuous Probability of Delay P(Delay), Delay Risk Tiers, and Actionable Recommendations.
Strictly handles missing values: genuine blank cells are preserved wherever data is absent.
Utilizes high-speed streaming xlsxwriter to export comprehensive multi-sheet executive Excel reports.
"""
import os
import joblib
import numpy as np
import pandas as pd
import xlsxwriter

from src.models import DELAY_TIER_NAMES, DELAY_ACTION_MAP


FEATURE_DESCRIPTIONS = {
    'planned_duration_days': '(Total planned manufacturing window in days = Basic Finish [GLTRP] minus Basic Start [GSTRP])',
    'creation_to_start_lead_days': '(Planning lead buffer in days from order creation [AUFK.ERDAT] to planned start date)',
    'release_delay_days': '(Execution release slip in days between actual shop floor release [FTRMI] and planned start)',
    'scheduled_vs_basic_slack_days': '(Scheduling float buffer in days between basic finish and detailed scheduled finish [GLTRS])',
    'total_order_qty_log': '(Log-transformed planned production batch volume [AFKO.GAMNG])',
    'planned_scrap_ratio': '(Expected scrap proportion planned in production bill = Planned Scrap / Total Order Qty)',
    'delivered_qty_ratio': '(Actual goods receipt fulfillment ratio = Delivered Qty [WEMNG] / Target Qty [PSMNG])',
    'unfulfilled_qty_log': '(Log-transformed unfulfilled quantity backlog remaining to be produced)',
    'confirmed_yield_ratio': '(Shop floor yield completion ratio = Confirmed Yield [AFRU.LMNGA] / Planned Order Qty)',
    'confirmation_scrap_rate': '(Confirmed scrap defect rate across operations = Confirmed Scrap / Total Confirmed Output)',
    'confirmation_count_log': '(Log-transformed total number of operation confirmation events logged in AFRU)',
    'confirmation_duration_days': '(Calendar days elapsed between first and final operation confirmations)',
    'confirmation_intensity': '(Confirmation logging velocity per planned manufacturing day)',
    'is_zero_confirmation': '(Binary indicator for production orders with zero confirmations logged)',
    'delivery_completed_flag_numeric': '(Binary indicator for final delivery completion flag [AFPO.ELIKZ])',
    'has_batch_flag': '(Binary indicator for batch-managed production orders [AFPO.CHARG])',
    'planned_deliv_lead_days': '(Material Master vendor/replenishment lead time in days [MARC.PLIFZ])',
    'plant_safety_stock_log': '(Log-transformed plant safety stock buffer for parent assembly [MARC.EISBE])',
    'net_weight_log': '(Log-transformed physical assembly net weight in kg [MARA.NTGEW])',
    'order_month': '(Calendar month of planned production start representing operational seasonality)',
    'order_quarter': '(Fiscal/Calendar quarter of production order execution)',
    'schedule_compression_index': '(Production load density index = Order Batch Size / Planned Duration Days)',
    'scrap_severity_index': '(Composite defect strain combining planned scrap and actual confirmed scrap)',
    'lead_time_strain': '(Ratio of planning buffer to manufacturing duration indicating dispatch urgency)',
    'order_type_encoded': '(Frequency-weighted encoding of SAP Production Order Type [AUFK.AUART])',
    'plant_encoded': '(Frequency-weighted encoding of manufacturing facility plant code [AUFK.WERKS])',
    'mrp_controller_encoded': '(Frequency-weighted encoding of Material Requirements Planner code [AFKO.DISPO])',
    'supervisor_encoded': '(Frequency-weighted encoding of shop-floor production supervisor [AFKO.FEVOR])',
    'material_type_encoded': '(Frequency-weighted encoding of SAP Material Master type [MARA.MTART])',
    'composite_delay_signal': '(Multi-criteria composite operational risk indicator combining release lag, scrap, and unfulfilled backlog)'
}

IGNORED_COLUMNS_INFO = [
    ("Order Number Primary Key", "AUFNR", "Unique alphanumeric SAP Order identifier. Primary key carrying zero intrinsic mathematical signal on manufacturing schedule slip."),
    ("Material Number", "MATNR", "Specific catalog inventory code. Abstracted into material type, weight, safety stock, and lead time attributes."),
    ("Order Text Description", "KTEXT", "Free-text description string retained strictly for human shop-floor identification and executive inspection."),
    ("Material Description", "MAKTX", "Descriptive textual name of finished/semi-finished good. Retained for executive dashboard readability."),
    ("Company Code", "BUKRS", "Corporate legal financial entity (e.g., 1000). Constant across operations with no predictive variance for machine delays."),
    ("Raw Calendar Timestamps", "ERDAT, AEDAT, GLTRP, GSTRP, GLTRI, IDAT2", "Calendar timestamps converted into relative lead times, durations, and seasonal indicators for invariant generalization."),
    ("Confirmation Internal IDs", "RUECK, RMZHL, BUDAT", "Sequential transactional confirmation line IDs aggregated into total yield, scrap, and event frequencies."),
    ("Storage Location Code", "LGORT", "Warehouse bin/storage staging code with high sparsity, abstracted into plant-level features.")
]


def score_with_single_model(
    champion_segmenter,
    df_featured: pd.DataFrame,
    scaler,
    feature_cols: list
) -> pd.DataFrame:
    """
    Applies the deployed #1 Champion Model to score all production order records.
    Assigns continuous P(Delay), binary prediction flags, risk tiers, and primary root causes.
    """
    df = df_featured.copy()
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    res = champion_segmenter.predict(X_scaled)
    df['predicted_delay_probability'] = np.round(res['scores'], 4)
    df['predicted_delay_flag'] = res['preds']
    df['delay_prediction_label'] = np.where(res['preds'] == 1, "DELAYED", "ON-TIME")
    df['delay_risk_tier'] = [DELAY_TIER_NAMES.get(t, f"Tier {t}") for t in res['mapped_tiers']]
    df['recommended_operational_action'] = [DELAY_ACTION_MAP.get(t, "Standard Monitoring") for t in res['mapped_tiers']]

    # Determine primary delay driver for explainability
    drivers = []
    for _, row in df.iterrows():
        if row['predicted_delay_flag'] == 0:
            drivers.append("Normal Production Cadence - No Significant Operational Bottleneck")
        else:
            if row['release_delay_days'] > 5.0:
                drivers.append(f"Shop Floor Release Lag ({row['release_delay_days']:.1f} days late release)")
            elif row['delivered_qty_ratio'] < 0.50 and row['total_order_qty'] > 500.0:
                drivers.append(f"Severe Fulfillment Backlog ({row['delivered_qty_ratio']*100.0:.1f}% delivered)")
            elif row['scrap_severity_index'] > 0.15:
                drivers.append(f"Excessive Scrap Rate ({row['scrap_severity_index']*100.0:.1f}% defect intensity)")
            elif row['is_zero_confirmation'] == 1.0:
                drivers.append("Zero Milestone Confirmations Logged in System")
            elif row['schedule_compression_index'] > 5.0:
                drivers.append("Heavy Batch Workload Relative to Allocated Time Window")
            else:
                drivers.append("Cumulative Lead Time Strain & Operational Bottlenecks")
    df['primary_delay_driver'] = drivers

    return df


def export_enriched_excel_report(
    scored_df: pd.DataFrame,
    output_path: str = "ML_7_Production_Order_Delay_Prediction_Enriched_Output_Clean.xlsx",
    fi_df: pd.DataFrame = None,
    comp_df: pd.DataFrame = None
):
    """
    Exports a professional, beautifully styled multi-sheet Excel workbook using xlsxwriter.
    Handles missing values: if absent, writes genuine blank cells.
    """
    print(f"[REPORT] Generating multi-sheet executive Excel workbook with xlsxwriter: {output_path}...")

    main_cols = [
        'order_number', 'order_type', 'plant', 'order_description', 'material_number', 'material_name',
        'planned_start_date', 'planned_finish_date', 'actual_completion_formatted', 'actual_delay_days_display',
        'total_order_qty', 'base_uom', 'item_delivered_qty', 'delivered_qty_ratio',
        'total_confirmations', 'total_conf_yield', 'total_conf_scrap', 'scrap_severity_index',
        'is_delayed_exception', 'predicted_delay_probability', 'predicted_delay_flag',
        'delay_prediction_label', 'delay_risk_tier', 'primary_delay_driver', 'recommended_operational_action'
    ]
    # Rename columns for clean reporting display
    rename_map = {
        'actual_completion_formatted': 'actual_completion_date',
        'actual_delay_days_display': 'actual_delay_days'
    }

    avail_cols = [c for c in main_cols if c in scored_df.columns]
    export_main = scored_df[avail_cols].copy().rename(columns=rename_map)

    # Delayed Watchlist (Tier 2 & Tier 3 orders)
    watchlist_df = export_main[
        export_main['delay_risk_tier'].isin([
            "Tier 2: Moderate Delay Risk / Bottlenecked",
            "Tier 3: Critical Delay Exception / Past Due"
        ])
    ].sort_values(by='predicted_delay_probability', ascending=False)

    workbook = xlsxwriter.Workbook(output_path, {'constant_memory': True})

    # Formats
    header_format = workbook.add_format({
        'bold': True,
        'font_name': 'Calibri',
        'font_size': 11,
        'font_color': '#FFFFFF',
        'bg_color': '#1B365D',
        'align': 'center',
        'valign': 'vcenter',
        'border': 1,
        'border_color': '#D9D9D9'
    })
    
    data_format = workbook.add_format({
        'font_name': 'Calibri',
        'font_size': 10,
        'valign': 'vcenter',
        'border': 1,
        'border_color': '#E0E0E0'
    })

    delayed_label_format = workbook.add_format({
        'bold': True,
        'font_name': 'Calibri',
        'font_size': 10,
        'font_color': '#9C0006',
        'bg_color': '#FFC7CE',
        'align': 'center',
        'valign': 'vcenter',
        'border': 1,
        'border_color': '#E0E0E0'
    })

    ontime_label_format = workbook.add_format({
        'bold': True,
        'font_name': 'Calibri',
        'font_size': 10,
        'font_color': '#006100',
        'bg_color': '#C6EFCE',
        'align': 'center',
        'valign': 'vcenter',
        'border': 1,
        'border_color': '#E0E0E0'
    })

    critical_tier_format = workbook.add_format({
        'bold': True,
        'font_name': 'Calibri',
        'font_size': 10,
        'font_color': '#9C0006',
        'bg_color': '#FFC7CE',
        'valign': 'vcenter',
        'border': 1,
        'border_color': '#E0E0E0'
    })

    moderate_tier_format = workbook.add_format({
        'bold': True,
        'font_name': 'Calibri',
        'font_size': 10,
        'font_color': '#9C6500',
        'bg_color': '#FFEB9C',
        'valign': 'vcenter',
        'border': 1,
        'border_color': '#E0E0E0'
    })

    low_tier_format = workbook.add_format({
        'font_name': 'Calibri',
        'font_size': 10,
        'bg_color': '#E6F0FA',
        'valign': 'vcenter',
        'border': 1,
        'border_color': '#E0E0E0'
    })

    ontime_tier_format = workbook.add_format({
        'bold': True,
        'font_name': 'Calibri',
        'font_size': 10,
        'font_color': '#006100',
        'bg_color': '#C6EFCE',
        'valign': 'vcenter',
        'border': 1,
        'border_color': '#E0E0E0'
    })

    prob_format = workbook.add_format({
        'font_name': 'Calibri',
        'font_size': 10,
        'align': 'right',
        'valign': 'vcenter',
        'num_format': '0.0000',
        'border': 1,
        'border_color': '#E0E0E0'
    })

    num_format = workbook.add_format({
        'font_name': 'Calibri',
        'font_size': 10,
        'align': 'right',
        'valign': 'vcenter',
        'num_format': '#,##0.00',
        'border': 1,
        'border_color': '#E0E0E0'
    })

    int_format = workbook.add_format({
        'font_name': 'Calibri',
        'font_size': 10,
        'align': 'right',
        'valign': 'vcenter',
        'num_format': '#,##0',
        'border': 1,
        'border_color': '#E0E0E0'
    })

    def write_sheet(sheet_title, df_data):
        ws = workbook.add_worksheet(sheet_title)
        ws.set_default_row(19)
        ws.set_row(0, 26)

        # Write Headers
        for col_idx, col_name in enumerate(df_data.columns):
            ws.write(0, col_idx, col_name, header_format)
            # Estimate width
            max_len = max(len(str(col_name)), 12)
            ws.set_column(col_idx, col_idx, min(max_len + 3, 38))

        # Write Rows with Missing Value Handling (BLANK if null/absent)
        for row_idx, row_vals in enumerate(df_data.values, 1):
            for col_idx, val in enumerate(row_vals):
                col_name = df_data.columns[col_idx]

                # Check if missing/blank
                if pd.isna(val) or val is None or val == '' or str(val).strip() in ['None', 'nan', 'UNKNOWN', 'NaN', 'null']:
                    ws.write_blank(row_idx, col_idx, None, data_format)
                elif col_name == 'delay_prediction_label':
                    if str(val) == 'DELAYED':
                        ws.write_string(row_idx, col_idx, 'DELAYED', delayed_label_format)
                    else:
                        ws.write_string(row_idx, col_idx, 'ON-TIME', ontime_label_format)
                elif col_name == 'delay_risk_tier':
                    s_val = str(val)
                    if 'Tier 3' in s_val or 'Critical' in s_val:
                        ws.write_string(row_idx, col_idx, s_val, critical_tier_format)
                    elif 'Tier 2' in s_val or 'Moderate' in s_val:
                        ws.write_string(row_idx, col_idx, s_val, moderate_tier_format)
                    elif 'Tier 1' in s_val or 'Low' in s_val:
                        ws.write_string(row_idx, col_idx, s_val, low_tier_format)
                    else:
                        ws.write_string(row_idx, col_idx, s_val, ontime_tier_format)
                elif 'probability' in col_name.lower():
                    try:
                        ws.write_number(row_idx, col_idx, float(val), prob_format)
                    except Exception:
                        ws.write_string(row_idx, col_idx, str(val), data_format)
                elif isinstance(val, (float, np.floating)):
                    if 'ratio' in col_name.lower() or 'rate' in col_name.lower() or 'index' in col_name.lower():
                        ws.write_number(row_idx, col_idx, float(val), prob_format)
                    else:
                        ws.write_number(row_idx, col_idx, float(val), num_format)
                elif isinstance(val, (int, np.integer)):
                    ws.write_number(row_idx, col_idx, int(val), int_format)
                else:
                    ws.write_string(row_idx, col_idx, str(val), data_format)

    # 1. Enriched Production Orders
    write_sheet("Enriched Production Orders", export_main)

    # 2. Delayed Orders Watchlist
    write_sheet("Delayed Orders Watchlist", watchlist_df)

    # 3. Feature Importance
    if fi_df is not None:
        fi_export = fi_df.copy()
        fi_export['Feature_Description'] = fi_export['Feature'].map(FEATURE_DESCRIPTIONS).fillna("")
        write_sheet("Feature Importance", fi_export)

    # 4. Model Comparison & Ranking
    if comp_df is not None:
        write_sheet("Model Comparison & Ranking", comp_df)

    # 5. Ignored Columns
    ign_df = pd.DataFrame(IGNORED_COLUMNS_INFO, columns=["Feature / Entity Name", "SAP Column Name", "Domain Exclusion Rationale"])
    write_sheet("Ignored Columns & Rationale", ign_df)

    workbook.close()
    print(f"[REPORT] Successfully generated and closed Excel workbook: {output_path}")


def save_single_model_pipeline(
    champion_segmenter,
    scaler,
    feature_cols: list,
    output_dir: str = "models"
):
    """Persists model and pipeline artifacts to disk."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "champion_delay_model.joblib")
    scaler_path = os.path.join(output_dir, "delay_scaler.joblib")
    bundle_path = os.path.join(output_dir, "delay_model_bundle.joblib")

    joblib.dump(champion_segmenter, model_path)
    joblib.dump(scaler, scaler_path)

    bundle = {
        'model_name': champion_segmenter.model_name,
        'segmenter': champion_segmenter,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'tier_names': DELAY_TIER_NAMES,
        'action_map': DELAY_ACTION_MAP,
        'threshold': champion_segmenter.threshold
    }
    joblib.dump(bundle, bundle_path)

    print(f"[MODEL] Champion Model persisted to: {model_path}")
    print(f"[MODEL] Preprocessing Scaler persisted to: {scaler_path}")
    print(f"[MODEL] Complete Deployment Bundle persisted to: {bundle_path}")
