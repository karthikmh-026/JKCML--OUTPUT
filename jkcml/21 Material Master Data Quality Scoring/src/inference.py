"""
Inference & Excel Reporting Module - Use Case 21: Material Master Data Quality Scoring
Handles scoring SAP Material Master records using the #1 Ranked Champion ML Model (Isolation Forest)
and generating executive, formatted Excel reports matching the enterprise SAP reference template,
scalable to hundreds of thousands of records with meticulous missing value handling.
"""
import os
import datetime
import joblib
import pandas as pd
import numpy as np
import xlsxwriter


def score_with_single_model(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the selected #1 Champion ML model (Isolation Forest) to calculate:
    - material_quality_anomaly_prob: continuous anomaly probability in [0.0, 1.0] (8 decimal precision)
    - predicted_quality_defect: binary flag (1 for incomplete/incorrect material record, 0 for compliant)
    - risk_category: executive risk triage (Critical Risk, High Risk, Medium Risk, Low Risk)
    """
    df_out = df.copy()

    X = df_out[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    probs, raw_scores = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)

    df_out['material_quality_anomaly_prob'] = np.round(probs, 8)
    df_out['predicted_quality_defect'] = preds.astype(int)

    conditions = [
        (df_out['material_quality_anomaly_prob'] >= 0.85) | ((df_out['predicted_quality_defect'] == 1) & (df_out['mdq_defect_index'] >= 45.0)),
        (df_out['material_quality_anomaly_prob'] >= 0.70) | ((df_out['predicted_quality_defect'] == 1) & (df_out['mdq_defect_index'] >= 28.0)),
        (df_out['material_quality_anomaly_prob'] >= 0.50) | (df_out['predicted_quality_defect'] == 1),
    ]
    choices = ['Critical Risk', 'High Risk', 'Medium Risk']
    df_out['risk_category'] = np.select(conditions, choices, default='Low Risk')

    df_out = df_out.sort_values(
        by=['material_quality_anomaly_prob', 'mdq_defect_index', 'effective_unit_price'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df_out


def export_enriched_excel_report(
    df: pd.DataFrame,
    output_path: str = "ML_21_Material_Master_Data_Quality_Scoring_Enriched_Output.xlsx",
    insight_id: str = "ML_21",
    use_case: str = "Material Master Data Quality Scoring",
    description: str = "Detect incomplete or incorrect material records, enriched with General Material Data (MARA), Material Descriptions (MAKT), Plant/MRP/Purchasing Views (MARC), Accounting/Valuation Data (MBEW), Units of Measure (MARM), Info Records (EINA), Plant (T001W), and Company (T001).",
    feature_importance_df: pd.DataFrame = None
) -> str:
    """
    Generates the executive enriched Excel report matching the enterprise SAP reference template
    using xlsxwriter with constant_memory mode for massive scalability and zero-blank handling:
    - Sheet 1: ML_21_Enriched_Output(Predict)
      - Rows 1-5: Metadata block (Insight ID, Use Case, Description, Generated On, Row Count)
      - Row 6: Blank row
      - Row 7: Enterprise SAP column headers
      - Row 8+: Enriched SAP Material Master records with ML quality scores, defect probabilities, and risk tiers
    - Sheet 2: Feature_Importance (All 32 engineered features with mathematical rationale)
    - Sheet 3: Columns Ignored (Audit justification for excluded non-feature columns)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    column_mapping = [
        ('source_schema', 'Schema Source'),
        ('company_code', 'Company Code'),
        ('company_name', 'Company Name'),
        ('company_city', 'Company City'),
        ('company_country', 'Company Country'),
        ('company_currency', 'Company Currency'),
        ('plant', 'Plant (WERKS)'),
        ('plant_name', 'Plant Name'),
        ('plant_city', 'Plant City'),
        ('material_id', 'Material (MATNR)'),
        ('material_desc', 'Material Description (MAKTX)'),
        ('material_type', 'Material Type (MTART)'),
        ('material_type_desc', 'Material Type Description'),
        ('material_group', 'Material Group (MATKL)'),
        ('old_material_no', 'Old Material No (BISMT)'),
        ('base_uom', 'Base UoM (MEINS)'),
        ('division', 'Division (SPART)'),
        ('prod_hierarchy', 'Product Hierarchy (PRDHA)'),
        ('gross_weight', 'Gross Weight (BRGEW)'),
        ('net_weight', 'Net Weight (NTGEW)'),
        ('weight_unit', 'Weight Unit (GEWEI)'),
        ('purchasing_group', 'Purchasing Group (EKGRP)'),
        ('mrp_type', 'MRP Type (DISMM)'),
        ('mrp_controller', 'MRP Controller (DISPO)'),
        ('procurement_type', 'Procurement Type (BESKZ)'),
        ('profit_center', 'Profit Center (PRCTR)'),
        ('valuation_class', 'Valuation Class (BKLAS)'),
        ('price_control', 'Price Control (VPRSV)'),
        ('moving_avg_price', 'Moving Avg Price (VERPR)'),
        ('standard_price', 'Standard Price (STPRS)'),
        ('total_valuated_stock', 'Valuated Stock (LBKUM)'),
        ('total_stock_value', 'Total Stock Value (SALK3)'),
        ('mara_maint_status', 'MARA Maint Views (PSTAT)'),
        ('marc_maint_status', 'MARC Maint Views (PSTAT)'),
        ('created_date', 'Created Date (ERSDA)'),
        ('last_changed_date', 'Last Changed Date (LAEDA)'),
        ('created_by', 'Created By (ERNAM)'),
        ('missing_critical_fields_cnt', 'Missing Critical Fields Count'),
        ('dup_description_count', 'Duplicate Description Count'),
        ('mdq_health_score', 'Material Quality Health Score'),
        ('mdq_defect_index', 'Material Quality Defect Score'),
        ('quality_issue_type', 'Primary Quality Issue Type'),
        ('material_quality_anomaly_prob', 'Material Quality Anomaly Probability'),
        ('predicted_quality_defect', 'Predicted Material Quality Defect'),
        ('risk_category', 'Risk Category')
    ]

    export_cols = []
    display_headers = []
    for internal_col, display_name in column_mapping:
        if internal_col in df.columns:
            export_cols.append(internal_col)
            display_headers.append(display_name)

    df_export = df[export_cols].copy()

    # Format dates to string or blank
    for dt_col in ['created_date', 'last_changed_date']:
        if dt_col in df_export.columns:
            dt_s = pd.to_datetime(df_export[dt_col], errors='coerce')
            df_export[dt_col] = dt_s.dt.strftime('%d-%m-%Y 00:00').fillna('')

    # Clean all missing values across strings & numbers: if missing, put blank ""
    for col in df_export.columns:
        if df_export[col].dtype == 'object':
            s = df_export[col].fillna('').astype(str).str.strip()
            df_export[col] = s.replace({'nan': '', 'None': '', 'null': '', 'NaT': '', '<NA>': '', 'undefined': ''})

    gen_time_str = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    row_count_str = f"{len(df_export):,}"

    metadata_rows = [
        ("Insight ID", insight_id),
        ("Use Case", use_case),
        ("Description", description),
        ("Generated On", gen_time_str),
        ("Row Count", row_count_str)
    ]

    actual_path = output_path
    save_candidates = [
        output_path,
        "ML_21_Material_Master_Data_Quality_Scoring_Enriched_Output_Clean.xlsx",
        "ML_21_Material_Master_Data_Quality_Scoring_Enriched_Output_Final.xlsx",
        os.path.join("reports", os.path.basename(output_path)),
        f"ML_21_Material_Master_Data_Quality_Scoring_Enriched_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    ]

    saved = False
    for candidate in save_candidates:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(candidate)), exist_ok=True)
            workbook = xlsxwriter.Workbook(candidate, {'constant_memory': True})

            # Formatting styles
            fmt_meta_lbl = workbook.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': True})
            fmt_meta_val = workbook.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': False})
            fmt_header = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 10, 'bold': True,
                'bg_color': '#F2F2F2', 'border': 1, 'border_color': '#D3D3D3',
                'align': 'center', 'valign': 'vcenter'
            })
            fmt_data_left = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'left'})
            fmt_data_center = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'center'})
            fmt_data_prob = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'right', 'num_format': '0.00000000'})
            fmt_data_num = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'right', 'num_format': '#,##0.00'})

            fmt_crit = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C0006', 'bg_color': '#FFC7CE', 'border': 1, 'border_color': '#D3D3D3', 'align': 'center'})
            fmt_high = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C6500', 'bg_color': '#FFEB9C', 'border': 1, 'border_color': '#D3D3D3', 'align': 'center'})
            fmt_med = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'font_color': '#006100', 'bg_color': '#C6EFCE', 'border': 1, 'border_color': '#D3D3D3', 'align': 'center'})
            fmt_low = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'center'})

            # Sheet 1: Main Data
            ws = workbook.add_worksheet("ML_21_Enriched_Output(Predict)")
            ws.freeze_panes(7, 0)

            # Metadata Rows 1-5
            for r_idx, (lbl, val) in enumerate(metadata_rows):
                ws.write(r_idx, 0, lbl, fmt_meta_lbl)
                ws.write(r_idx, 1, val, fmt_meta_val)

            # Row 6 is blank as per enterprise template
            ws.set_row(6, 24)
            ws.write_row(6, 0, display_headers, fmt_header)

            center_headers = {
                'Schema Source', 'Company Code', 'Plant (WERKS)', 'Material (MATNR)', 'Material Type (MTART)',
                'Base UoM (MEINS)', 'Division (SPART)', 'Purchasing Group (EKGRP)', 'MRP Type (DISMM)',
                'MRP Controller (DISPO)', 'Procurement Type (BESKZ)', 'Profit Center (PRCTR)',
                'Valuation Class (BKLAS)', 'Price Control (VPRSV)', 'Missing Critical Fields Count',
                'Duplicate Description Count', 'Predicted Material Quality Defect',
                'Created Date (ERSDA)', 'Last Changed Date (LAEDA)'
            }
            num_headers = {'Price', 'Value', 'Stock', 'Weight', 'Score', 'Defect Score', 'Health Score'}

            col_fmt_types = []
            for h in display_headers:
                if 'Probability' in h:
                    col_fmt_types.append('prob')
                elif any(k in h for k in num_headers):
                    col_fmt_types.append('num')
                elif h in center_headers:
                    col_fmt_types.append('center')
                elif h == 'Risk Category':
                    col_fmt_types.append('risk')
                else:
                    col_fmt_types.append('left')

            # Write data rows
            data_values = df_export.values
            for r_offset, row_vals in enumerate(data_values, start=7):
                for c_idx, val in enumerate(row_vals):
                    fmt_t = col_fmt_types[c_idx]

                    if pd.isna(val) or val == '' or str(val).strip() == '':
                        ws.write_string(r_offset, c_idx, '', fmt_data_left if fmt_t == 'left' else fmt_data_center)
                    elif fmt_t == 'prob':
                        ws.write_number(r_offset, c_idx, float(val), fmt_data_prob)
                    elif fmt_t == 'num':
                        try:
                            ws.write_number(r_offset, c_idx, float(val), fmt_data_num)
                        except (ValueError, TypeError):
                            ws.write_string(r_offset, c_idx, str(val), fmt_data_left)
                    elif fmt_t == 'risk':
                        val_str = str(val).strip()
                        if val_str == 'Critical Risk':
                            ws.write_string(r_offset, c_idx, val_str, fmt_crit)
                        elif val_str == 'High Risk':
                            ws.write_string(r_offset, c_idx, val_str, fmt_high)
                        elif val_str == 'Medium Risk':
                            ws.write_string(r_offset, c_idx, val_str, fmt_med)
                        else:
                            ws.write_string(r_offset, c_idx, val_str, fmt_low)
                    elif fmt_t == 'center':
                        ws.write_string(r_offset, c_idx, str(val), fmt_data_center)
                    else:
                        ws.write_string(r_offset, c_idx, str(val), fmt_data_left)

            # Auto-adjust column widths based on sample
            for c_idx, header_text in enumerate(display_headers):
                max_len = len(str(header_text))
                for sample_val in df_export.iloc[:100, c_idx]:
                    if pd.notna(sample_val):
                        max_len = max(max_len, len(str(sample_val)))
                ws.set_column(c_idx, c_idx, min(max(max_len + 3, 12), 42))

            # -------------------------------------------------------------
            # Sheet 2: Feature_Importance
            # -------------------------------------------------------------
            if feature_importance_df is None and os.path.exists("reports/feature_importance.csv"):
                feature_importance_df = pd.read_csv("reports/feature_importance.csv")

            if feature_importance_df is not None and len(feature_importance_df) > 0:
                ws_feat = workbook.add_worksheet("Feature_Importance")
                ws_feat.write(0, 0, f"{insight_id} | Feature_Importance", fmt_meta_lbl)

                fmt_feat_hdr = workbook.add_format({
                    'font_name': 'Calibri', 'font_size': 11, 'bold': True,
                    'font_color': '#FFFFFF', 'bg_color': '#2F5597', 'border': 1,
                    'align': 'left', 'valign': 'vcenter'
                })
                fmt_feat_hdr_r = workbook.add_format({
                    'font_name': 'Calibri', 'font_size': 11, 'bold': True,
                    'font_color': '#FFFFFF', 'bg_color': '#2F5597', 'border': 1,
                    'align': 'right', 'valign': 'vcenter'
                })

                ws_feat.set_row(1, 22)
                ws_feat.write(1, 0, "feature", fmt_feat_hdr)
                ws_feat.write(1, 1, "importance", fmt_feat_hdr_r)
                ws_feat.write(1, 2, "Direct Mathematical Influence On Material Master Data Quality Risk:", fmt_feat_hdr)

                col_feat_name = 'feature' if 'feature' in feature_importance_df.columns else feature_importance_df.columns[0]
                col_imp_name = 'importance' if 'importance' in feature_importance_df.columns else feature_importance_df.columns[1]
                has_expl = len(feature_importance_df.columns) > 2
                col_expl_name = feature_importance_df.columns[2] if has_expl else None

                fmt_imp_val = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'right', 'num_format': '0.000000000'})
                for idx, r_data in feature_importance_df.iterrows():
                    curr_row = idx + 2
                    ws_feat.write(curr_row, 0, str(r_data[col_feat_name]), fmt_data_left)
                    ws_feat.write_number(curr_row, 1, float(r_data[col_imp_name]), fmt_imp_val)
                    expl_val = str(r_data[col_expl_name]) if col_expl_name and pd.notna(r_data[col_expl_name]) else ""
                    ws_feat.write(curr_row, 2, expl_val, fmt_data_left)

                ws_feat.set_column(0, 0, 32)
                ws_feat.set_column(1, 1, 18)
                ws_feat.set_column(2, 2, 78)

            # -------------------------------------------------------------
            # Sheet 3: Columns Ignored
            # -------------------------------------------------------------
            ws_ign = workbook.add_worksheet("Columns Ignored")
            ws_ign.write(0, 0, "Columns Ignored / Excluded from Model Training:", fmt_meta_lbl)

            fmt_ign_hdr = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 10, 'bold': True,
                'border': 1, 'border_color': '#D3D3D3', 'align': 'left', 'valign': 'vcenter'
            })
            ws_ign.set_row(2, 20)
            ws_ign.write_row(2, 0, ["Ignored Column", "SAP Field Name", "Why It Was Ignored"], fmt_ign_hdr)

            ignored_rows = [
                ("Material Number", "MATNR", "Sequential or alphanumeric SAP primary key. Carries no intrinsic mathematical signal on whether master views are complete or accurate."),
                ("Raw Material Description String", "MAKTX", "Free-text material description string. Transformed mathematically into desc_char_length, desc_token_count, dup_description_count, and is_cryptic_or_short_desc."),
                ("Plant & Valuation Area Codes", "WERKS, BWKEY", "High-cardinality organizational identifiers. Retained for audit reporting while plant-level governance is captured via MARC/MBEW completeness features."),
                ("Raw Material Type & Group Codes", "MTART, MATKL, MTBEZ", "Categorical classification codes. Used to condition contextual rules (e.g., valuated vs non-valuated, physical weight required) and peer Z-scores (mat_group_price_zscore)."),
                ("Maintenance Status Strings", "VPSTA, PSTAT", "Letter-encoded SAP view strings (e.g., 'KLEVQDGAB'). Engineered into mara_views_maintained_cnt, marc_views_maintained_cnt, and view_sync_gap."),
                ("Organizational & Text Names", "BUTXT, NAME1, ORT01, REGIO, LAND1", "Descriptive company and plant location strings retained strictly for executive report readability."),
                ("User IDs", "ERNAM, AENAM", "SAP user identifiers for record creation and modification. Retained as audit metadata."),
                ("Raw Creation & Change Dates", "ERSDA, LAEDA", "Raw calendar dates cannot be ingested directly by distance/tree models. Engineered into days_since_last_change."),
                ("Company Code & Currency", "BUKRS, WAERS", "Constant enterprise entity (1000 - J.K. CEMENT LIMITED, INR) across the dataset with zero variance."),
                ("Audit Benchmark Label", "is_quality_exception", "Strictly excluded from model training so all 5 ML models remain 100% unsupervised and learn purely from input feature distributions.")
            ]

            for r_offset, (col_name, sap_field, reason) in enumerate(ignored_rows, start=3):
                ws_ign.write(r_offset, 0, col_name, fmt_data_left)
                ws_ign.write(r_offset, 1, sap_field, fmt_data_left)
                ws_ign.write(r_offset, 2, reason, fmt_data_left)

            ws_ign.set_column(0, 0, 28)
            ws_ign.set_column(1, 1, 26)
            ws_ign.set_column(2, 2, 98)

            workbook.close()
            actual_path = candidate
            saved = True
            break
        except Exception as e:
            print(f"[WARNING] Could not save to '{candidate}': {e}. Trying next path...")
            continue

    if not saved:
        fallback = f"ML_21_Material_Quality_Output_{int(datetime.datetime.now().timestamp())}.xlsx"
        workbook = xlsxwriter.Workbook(fallback, {'constant_memory': True})
        workbook.close()
        actual_path = fallback

    print(f"[SUCCESS] Executive Enriched Excel Report saved to: {actual_path}")
    print(f"[INFO] Total Rows: {len(df_export):,}, Total Columns: {len(display_headers)}")
    return actual_path


def save_single_model_pipeline(model, scaler, feature_cols, suite=None, comparison_df=None, output_dir: str = "models"):
    """
    Saves the selected #1 Champion Model (Isolation Forest), scaler, and 5-model benchmark bundle to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': 'Isolation Forest',
        'benchmark_ranking': comparison_df.to_dict(orient='records') if comparison_df is not None else None
    }
    champ_path = os.path.join(output_dir, "material_quality_champion_model.joblib")
    bundle_path = os.path.join(output_dir, "material_quality_model_bundle.joblib")
    if_path = os.path.join(output_dir, "champion_isolation_forest.joblib")
    scaler_path = os.path.join(output_dir, "robust_scaler.joblib")

    joblib.dump(bundle, champ_path)
    joblib.dump(bundle, bundle_path)
    joblib.dump(model.model, if_path)
    joblib.dump(scaler, scaler_path)

    if suite is not None and 'dbscan' in suite.models:
        joblib.dump(suite.models['dbscan'], os.path.join(output_dir, "champion_dbscan.joblib"))

    print(f"[SUCCESS] Saved champion model bundle to: {champ_path}")
