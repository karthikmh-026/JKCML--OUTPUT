"""
Inference & Excel Reporting Module - Use Case 26: Breakdown Root Cause Analysis (NLP)
Handles scoring and extracting breakdown root causes from SAP Plant Maintenance texts using
the #1 Ranked Champion Unsupervised Clustering Model (K-Means Clustering) and generating executive
3-sheet Excel reports matching the enterprise SAP template.
"""
import os
import datetime
import joblib
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def score_with_single_model(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the selected #1 Champion NLP Clustering ML model (K-Means Clustering) to calculate:
    - rca_cluster_id: ordered cluster tier (0 = Preventive/Calibrated, 1 = Verified Corrective, 2 = Moderate/Chronic Wear, 3 = Critical Breakdown/Exception)
    - rca_segment_name: human-readable Breakdown Root-Cause Segment
    - breakdown_exception_prob: continuous Breakdown / Unverified RCA probability in [0.0, 1.0] (8 decimals)
    - predicted_breakdown_exception: binary flag (1 for Critical Breakdown / Unverified RCA Exception, 0 for Verified PM Order)
    - risk_category: executive triage tier (Critical Risk, High Risk, Medium Risk, Low Risk)
    """
    df_out = df.copy()

    X = df_out[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    probs, raw_scores, mapped_tiers = model.predict_proba(X_scaled)
    _, segment_names = model.predict_clusters(X_scaled)
    preds = model.predict(X_scaled)

    df_out['rca_cluster_id'] = mapped_tiers.astype(int)
    df_out['rca_segment_name'] = segment_names
    df_out['breakdown_exception_prob'] = np.round(probs, 8)
    df_out['predicted_breakdown_exception'] = preds.astype(int)

    conditions = [
        (df_out['rca_cluster_id'] == 3) & (
            (df_out['order_type'] == 'PM01') |
            (df_out['is_unclosed_or_high_cost_breakdown'] == 1) |
            (df_out['is_work_center_domain_mismatch'] == 1) |
            (df_out['breakdown_severity_risk_index'] >= 56.0)
        ),
        (df_out['rca_cluster_id'] == 3) | (df_out['breakdown_exception_prob'] >= 0.72),
        (df_out['rca_cluster_id'] == 2) | (df_out['breakdown_exception_prob'] >= 0.48),
    ]
    choices = ['Critical Risk', 'High Risk', 'Medium Risk']
    df_out['risk_category'] = np.select(conditions, choices, default='Low Risk')

    # Sort so highest breakdown severity & exception probability with highest spare-parts cost appear at the top
    df_out = df_out.sort_values(
        by=['breakdown_exception_prob', 'is_breakdown_or_emergency_order', 'total_spare_cost_inr'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df_out


def export_enriched_excel_report(
    df: pd.DataFrame,
    output_path: str = "ML_26_Breakdown_Root_Cause_Analysis_Enriched_Output.xlsx",
    insight_id: str = "ML_26",
    use_case: str = "Breakdown Root Cause Analysis",
    description: str = "Extract root causes from maintenance text using Unsupervised NLP (TF-IDF + LSA Semantic Embeddings + Lexical Failure-Mode Taxonomy) & Clustering across SAP Plant Maintenance Order Master (AUFK), Order Header (AFKO), Order Confirmations (AFRU), Spare Parts Goods Movements (AUFM, MAKT), Component Reservations (RESB), and Work Centers (CRHD).",
    feature_importance_df: pd.DataFrame = None
) -> str:
    """
    Generates the executive enriched Excel report matching the enterprise SAP reference template:
    - Sheet 1: ML_26_Enriched_Output(Predict)
      - Rows 1-5: Metadata block (Insight ID, Use Case, Description, Generated On, Row Count)
      - Row 6: Blank row
      - Row 7: Enterprise SAP column headers
      - Row 8+: Enriched SAP Plant Maintenance records with NLP extracted equipment tag, extracted root cause code, root cause description, failure domain, cluster assignments, confidence scores, probabilities, and risk tiers
    - Sheet 2: Feature_Importance (All 32 engineered NLP & Plant Maintenance features with mathematical rationale)
    - Sheet 3: Columns Ignored (Audit justification for excluded non-feature columns)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    column_mapping = [
        ('company_code', 'Company Code (BUKRS)'),
        ('company_name', 'Company Name (BUTXT)'),
        ('order_number', 'Maintenance Order (AUFNR)'),
        ('order_type', 'Order Type (AUART)'),
        ('order_type_desc', 'Order Type Description'),
        ('plant', 'Plant (WERKS)'),
        ('plant_name', 'Plant Name (T001W)'),
        ('work_center', 'Work Center (VAPLZ)'),
        ('cost_center', 'Cost Center (KOSTV)'),
        ('profit_center', 'Profit Center (PRCTR)'),
        ('created_date', 'Created Date (ERDAT)'),
        ('teco_date', 'TECO Date (IDAT2)'),
        ('phase_released', 'Released (PHAS1)'),
        ('phase_teco', 'TECO Status (PHAS2)'),
        ('phase_closed', 'Closed Status (PHAS3)'),
        ('maintenance_text', 'Maintenance Order Text (AUFK-KTEXT)'),
        ('confirmation_text', 'Technician Confirmation Text (AFRU-LTXA1)'),
        ('spare_parts_texts', 'Consumed Spare Parts (AUFM/MAKT)'),
        ('extracted_equipment_tag', 'NLP Extracted Equipment Tag'),
        ('extracted_root_cause_code', 'NLP Root Cause Code'),
        ('extracted_root_cause', 'NLP Extracted Root Cause'),
        ('extracted_root_cause_domain', 'NLP Root Cause Engineering Domain'),
        ('maint_text_char_length', 'Maint Text Length (Chars)'),
        ('maint_text_word_count', 'Maint Text Word Count'),
        ('failure_symptom_token_count', 'Failure Symptom Tokens'),
        ('tfidf_active_ngrams', 'Active TF-IDF N-Grams'),
        ('equipment_order_frequency', 'Equipment Order Frequency'),
        ('confirmation_count', 'Confirmation Count (AFRU)'),
        ('total_actual_duration', 'Actual Duration (AFRU-IDAUR)'),
        ('distinct_spares_count', 'Distinct Spares Issued'),
        ('total_spare_cost_inr', 'Spare Parts Cost INR (AUFM-DMBTR)'),
        ('order_turnaround_days', 'Turnaround Days'),
        ('text_diagnostic_clarity_score', 'Diagnostic Clarity Score (0-100)'),
        ('rca_semantic_specificity_score', 'RCA Specificity Score (0-100)'),
        ('pm_execution_alignment_score', 'PM Execution Alignment (0-100)'),
        ('rca_tier_code', 'RCA Tier Code (D-R-E)'),
        ('nlp_rca_confidence_score', 'NLP RCA Confidence Score'),
        ('breakdown_severity_risk_index', 'Breakdown Severity Risk Index'),
        ('rca_cluster_id', 'ML Cluster ID'),
        ('rca_segment_name', 'ML Breakdown RCA Segment'),
        ('breakdown_exception_prob', 'Breakdown Exception Prob'),
        ('predicted_breakdown_exception', 'Predicted Breakdown Exception'),
        ('risk_category', 'Risk Category')
    ]

    export_cols = []
    display_headers = []
    for internal_col, display_name in column_mapping:
        if internal_col in df.columns:
            export_cols.append(internal_col)
            display_headers.append(display_name)

    df_export = df[export_cols].copy()

    # Sanitize missing values: if present format properly, if not present put blank ""
    for dt_col in ['created_date', 'teco_date']:
        if dt_col in df_export.columns:
            df_export[dt_col] = pd.to_datetime(df_export[dt_col], errors='coerce').dt.strftime('%d-%m-%Y').fillna('')

    for num_col in [
        'total_actual_duration', 'total_spare_cost_inr', 'text_diagnostic_clarity_score',
        'rca_semantic_specificity_score', 'pm_execution_alignment_score',
        'nlp_rca_confidence_score', 'breakdown_severity_risk_index'
    ]:
        if num_col in df_export.columns:
            df_export[num_col] = pd.to_numeric(df_export[num_col], errors='coerce').fillna(0.0).round(2)

    if 'breakdown_exception_prob' in df_export.columns:
        df_export['breakdown_exception_prob'] = pd.to_numeric(df_export['breakdown_exception_prob'], errors='coerce').fillna(0.0).round(8)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ML_26_Enriched_Output(Predict)"

    font_meta_label = Font(name="Calibri", size=11, bold=True, color="000000")
    font_meta_val = Font(name="Calibri", size=11, bold=False, color="000000")

    gen_time_str = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    row_count_str = f"{len(df_export):,}"

    metadata_rows = [
        ("Insight ID", insight_id),
        ("Use Case", use_case),
        ("Description", description),
        ("Generated On", gen_time_str),
        ("Row Count", row_count_str)
    ]

    for r_idx, (label, val) in enumerate(metadata_rows, start=1):
        cell_lbl = ws.cell(row=r_idx, column=1, value=label)
        cell_val = ws.cell(row=r_idx, column=2, value=val)
        cell_lbl.font = font_meta_label
        cell_val.font = font_meta_val

    # Row 6 left blank intentionally as per enterprise SAP template
    ws.cell(row=6, column=1, value="")

    header_row_idx = 7
    font_header = Font(name="Calibri", size=10, bold=True, color="000000")
    font_data = Font(name="Calibri", size=10, bold=False, color="000000")
    fill_header = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    thin_side = Side(border_style="thin", color="D3D3D3")
    border_header = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

    for col_idx, header_text in enumerate(display_headers, start=1):
        cell = ws.cell(row=header_row_idx, column=col_idx, value=header_text)
        cell.font = font_header
        cell.fill = fill_header
        cell.border = border_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=False)

    ws.row_dimensions[header_row_idx].height = 24

    # Ultra-fast bulk row export with complete missing value handling
    # "check the missing value handilngs of that data check properly and it is there put not there just put that blank"
    for row in df_export.itertuples(index=False):
        clean_row = []
        for val in row:
            if val is None or pd.isna(val):
                clean_row.append("")
            elif isinstance(val, (int, np.integer)):
                clean_row.append(int(val))
            elif isinstance(val, (float, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    clean_row.append("")
                else:
                    clean_row.append(float(val))
            elif isinstance(val, str):
                s = val.strip()
                if s.lower() in ('none', 'nan', 'nat', '<na>', 'null'):
                    clean_row.append("")
                else:
                    clean_row.append(s)
            else:
                clean_row.append(val)
        ws.append(clean_row)

    # Set column widths based on headers and sample of top 200 rows
    for c_idx, header_text in enumerate(display_headers, start=1):
        col_letter = get_column_letter(c_idx)
        max_len = len(str(header_text))
        for r_sample in range(8, min(108, len(df_export) + 8)):
            val = ws.cell(row=r_sample, column=c_idx).value
            if val is not None and val != "":
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, 11), 44)

    ws.freeze_panes = "A8"

    # -------------------------------------------------------------
    # Sheet 2: Feature_Importance
    # -------------------------------------------------------------
    if feature_importance_df is None and os.path.exists("reports/feature_importance.csv"):
        feature_importance_df = pd.read_csv("reports/feature_importance.csv")

    if feature_importance_df is not None and len(feature_importance_df) > 0:
        ws_feat = wb.create_sheet(title="Feature_Importance")

        cell_title = ws_feat.cell(row=1, column=1, value=f"{insight_id} | Feature_Importance")
        cell_title.font = Font(name="Calibri", size=11, bold=True, color="000000")

        c_feat = ws_feat.cell(row=2, column=1, value="feature")
        c_imp = ws_feat.cell(row=2, column=2, value="importance")
        c_expl = ws_feat.cell(row=2, column=3, value="Direct Mathematical Influence On Breakdown Root Cause Extraction & PM Governance:")

        font_feat_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        fill_feat_header = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")

        c_feat.font = font_feat_header
        c_feat.fill = fill_feat_header
        c_feat.border = border_header
        c_feat.alignment = Alignment(horizontal="left", vertical="center")

        c_imp.font = font_feat_header
        c_imp.fill = fill_feat_header
        c_imp.border = border_header
        c_imp.alignment = Alignment(horizontal="right", vertical="center")

        c_expl.font = font_feat_header
        c_expl.fill = fill_feat_header
        c_expl.border = border_header
        c_expl.alignment = Alignment(horizontal="left", vertical="center")

        ws_feat.row_dimensions[2].height = 22

        col_feat_name = 'feature' if 'feature' in feature_importance_df.columns else feature_importance_df.columns[0]
        col_imp_name = 'importance' if 'importance' in feature_importance_df.columns else feature_importance_df.columns[1]
        has_expl = len(feature_importance_df.columns) > 2
        col_expl_name = feature_importance_df.columns[2] if has_expl else None

        for idx, r_data in feature_importance_df.iterrows():
            curr_row = idx + 3
            cell_f = ws_feat.cell(row=curr_row, column=1, value=str(r_data[col_feat_name]))
            cell_f.font = font_data
            cell_f.border = border_header
            cell_f.alignment = Alignment(horizontal="left", vertical="center")

            imp_val = float(r_data[col_imp_name])
            cell_i = ws_feat.cell(row=curr_row, column=2, value=imp_val)
            cell_i.font = font_data
            cell_i.border = border_header
            cell_i.alignment = Alignment(horizontal="right", vertical="center")
            cell_i.number_format = "0.000000000"

            expl_val = str(r_data[col_expl_name]) if col_expl_name and pd.notna(r_data[col_expl_name]) else ""
            cell_e = ws_feat.cell(row=curr_row, column=3, value=expl_val)
            cell_e.font = font_data
            cell_e.border = border_header
            cell_e.alignment = Alignment(horizontal="left", vertical="center")

        ws_feat.column_dimensions['A'].width = 38
        ws_feat.column_dimensions['B'].width = 18
        ws_feat.column_dimensions['C'].width = 92

    # -------------------------------------------------------------
    # Sheet 3: Columns Ignored
    # -------------------------------------------------------------
    ws_ign = wb.create_sheet(title="Columns Ignored")

    cell_ign_title = ws_ign.cell(row=1, column=1, value="Columns Ignored / Excluded from Model Training:")
    cell_ign_title.font = Font(name="Calibri", size=11, bold=True, color="000000")

    ign_headers = ["Ignored Column", "SAP Field Name", "Why It Was Ignored"]
    for c_idx, h_text in enumerate(ign_headers, start=1):
        c_cell = ws_ign.cell(row=3, column=c_idx, value=h_text)
        c_cell.font = Font(name="Calibri", size=10, bold=True, color="000000")
        c_cell.border = border_header
        c_cell.alignment = Alignment(horizontal="left", vertical="center")

    ws_ign.row_dimensions[3].height = 20

    ignored_rows = [
        ("Maintenance Order Primary Keys", "AUFNR, RUECK, RMZHL, RSNUM", "Unique SAP PM order, confirmation, and reservation keys. Carry no intrinsic semantic signal on failure root causes."),
        ("Raw Unvectorized Maintenance Texts", "KTEXT, LTXA1, MAKTX, POTX1", "Raw text strings cannot be fed directly into Euclidean/Gaussian clustering; transformed into TF-IDF n-grams, 5 TruncatedSVD semantic components, and lexical failure-mode affinities."),
        ("Organizational & Cost Object Codes", "BUKRS, WERKS, KOSTV, PRCTR, GSBER", "Company code, plant, cost center, and profit center retained strictly for plant maintenance governance and audit drill-down."),
        ("Raw Calendar Scheduling Timestamps", "ERDAT, IDAT1, IDAT2, GSTRP, GLTRP", "Raw calendar dates cannot be clustered directly; converted into engineered order_turnaround_days and lifecycle status signals."),
        ("User & Supervisor Identifiers", "ERNAM, AENAM, DISPO, FEVOR, VERAN", "SAP user IDs and planner codes excluded to prevent user-specific bias in unsupervised root cause clustering."),
        ("Storage Location & Unit Codes", "LGORT, MEINS, IDAUE, WAERS", "Inventory storage locations and unit-of-measure codes retained for spare parts traceability."),
        ("Post-Hoc Benchmark Segment & Label", "benchmark_rca_segment, is_breakdown_rca_exception", "Strictly excluded from model training so all 5 NLP clustering models remain 100% unsupervised and extract root causes purely from input feature distributions.")
    ]

    for r_offset, (col_name, sap_field, reason) in enumerate(ignored_rows, start=4):
        c1 = ws_ign.cell(row=r_offset, column=1, value=col_name)
        c2 = ws_ign.cell(row=r_offset, column=2, value=sap_field)
        c3 = ws_ign.cell(row=r_offset, column=3, value=reason)

        for c_cell in [c1, c2, c3]:
            c_cell.font = font_data
            c_cell.border = border_header
            c_cell.alignment = Alignment(horizontal="left", vertical="center")

    ws_ign.column_dimensions['A'].width = 34
    ws_ign.column_dimensions['B'].width = 30
    ws_ign.column_dimensions['C'].width = 98

    actual_path = output_path
    save_candidates = [
        output_path,
        "ML_26_Breakdown_Root_Cause_Analysis_Enriched_Output_Clean.xlsx",
        "ML_26_Breakdown_Root_Cause_Analysis_Enriched_Output_Final.xlsx",
        os.path.join("reports", os.path.basename(output_path)),
        f"ML_26_Breakdown_Root_Cause_Analysis_Enriched_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    ]

    saved = False
    for candidate in save_candidates:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(candidate)), exist_ok=True)
            wb.save(candidate)
            actual_path = candidate
            saved = True
            break
        except PermissionError:
            print(f"[WARNING] '{candidate}' is currently open in Excel / locked. Trying next file...")
            continue

    if not saved:
        fallback = f"ML_26_Breakdown_RCA_Output_{int(datetime.datetime.now().timestamp())}.xlsx"
        wb.save(fallback)
        actual_path = fallback

    print(f"[SUCCESS] Executive Enriched Excel Report saved to: {actual_path}")
    print(f"[INFO] Total Rows: {len(df_export):,}, Total Columns: {len(display_headers)}, Sheets: {wb.sheetnames}")
    return actual_path


def save_single_model_pipeline(model, scaler, feature_cols, suite=None, comparison_df=None, output_dir: str = "models"):
    """
    Saves the selected #1 Champion NLP Clustering Model (K-Means Clustering), scaler, and 5-model benchmark bundle to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': 'K-Means Clustering',
        'benchmark_ranking': comparison_df.to_dict(orient='records') if comparison_df is not None else None
    }
    champ_path = os.path.join(output_dir, "breakdown_rca_champion_model.joblib")
    bundle_path = os.path.join(output_dir, "breakdown_rca_model_bundle.joblib")
    km_path = os.path.join(output_dir, "champion_kmeans.joblib")
    scaler_path = os.path.join(output_dir, "robust_scaler.joblib")

    joblib.dump(bundle, champ_path)
    joblib.dump(bundle, bundle_path)
    joblib.dump(model.model, km_path)
    joblib.dump(scaler, scaler_path)

    if suite is not None and 'gmm' in suite.models:
        joblib.dump(suite.models['gmm'], os.path.join(output_dir, "challenger_gmm.joblib"))

    print(f"[SUCCESS] Saved champion NLP clustering model bundle to: {champ_path}")
