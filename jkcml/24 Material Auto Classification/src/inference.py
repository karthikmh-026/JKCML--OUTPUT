"""
Inference & Excel Reporting Module - Use Case 24: Material Auto Classification (NLP)
Handles scoring and auto-classifying SAP Material records from descriptions using the #1 Ranked
Champion Unsupervised Clustering Model (K-Means Clustering) and generating executive 3-sheet
Excel reports matching the enterprise SAP template.
Includes comprehensive missing value handling (blanks for any missing/null values) and
high-performance streaming support for large-scale enterprise datasets.
"""
import os
import datetime
import joblib
import pandas as pd
import numpy as np
import xlsxwriter


def score_with_single_model(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the selected #1 Champion NLP Clustering ML model (K-Means Clustering) to calculate:
    - material_cluster_id: ordered cluster tier (0 = High-Confidence Core, 1 = Verified Technical Spares, 2 = Moderate-Confidence, 3 = Misclassified/Ambiguous)
    - material_segment_name: human-readable Material Auto-Classification Segment
    - classification_exception_prob: continuous misclassification/ambiguity probability in [0.0, 1.0] (8 decimals)
    - predicted_classification_exception: binary flag (1 for Misclassified / Ambiguous material exception, 0 for Verified Auto-Classified)
    - risk_category: executive triage tier (Critical Risk, High Risk, Medium Risk, Low Risk)
    """
    df_out = df.copy()

    X = df_out[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    probs, raw_scores, mapped_tiers = model.predict_proba(X_scaled)
    _, segment_names = model.predict_clusters(X_scaled)
    preds = model.predict(X_scaled)

    df_out['material_cluster_id'] = mapped_tiers.astype(int)
    df_out['material_segment_name'] = segment_names
    df_out['classification_exception_prob'] = np.round(probs, 8)
    df_out['predicted_classification_exception'] = preds.astype(int)

    conditions = [
        (df_out['material_cluster_id'] == 3) & (
            (df_out['is_nlp_mtart_conflict'] == 1) |
            (df_out['is_po_matkl_mismatch'] == 1) |
            (df_out['is_cryptic_or_short_desc'] == 1) |
            (df_out['nlp_misclassification_risk_index'] >= 76.0)
        ),
        (df_out['material_cluster_id'] == 3) | (df_out['classification_exception_prob'] >= 0.72),
        (df_out['material_cluster_id'] == 2) | (df_out['classification_exception_prob'] >= 0.48),
    ]
    choices = ['Critical Risk', 'High Risk', 'Medium Risk']
    df_out['risk_category'] = np.select(conditions, choices, default='Low Risk')

    # Sort so highest misclassification / ambiguity exceptions with spend/stock exposure appear at the top
    df_out = df_out.sort_values(
        by=['classification_exception_prob', 'is_nlp_mtart_conflict', 'total_po_spend', 'total_stock_value'],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    return df_out


def export_enriched_excel_report(
    df: pd.DataFrame,
    output_path: str = "ML_24_Material_Auto_Classification_Enriched_Output.xlsx",
    insight_id: str = "ML_24",
    use_case: str = "Material Auto Classification",
    description: str = "Automatically classify materials from descriptions using Unsupervised NLP (TF-IDF + LSA Semantic Embeddings + Lexical Taxonomy) & Clustering across SAP Material Master (MARA), Material Descriptions (MAKT), Plant/MRP (MARC), Valuation (MBEW), and Purchase Order Short Texts (EKPO).",
    feature_importance_df: pd.DataFrame = None
) -> str:
    """
    Generates the executive enriched Excel report matching the enterprise SAP reference template:
    - Sheet 1: ML_24_Enriched_Output(Predict)
      - Rows 1-5: Metadata block (Insight ID, Use Case, Description, Generated On, Row Count)
      - Row 6: Blank row
      - Row 7: Enterprise SAP column headers
      - Row 8+: Enriched SAP Material Auto-Classification records with NLP domains, recommended MATKL, cluster assignments, confidence scores, probabilities, and risk tiers
    - Sheet 2: Feature_Importance (All 32 engineered NLP & Material Classification features with mathematical rationale)
    - Sheet 3: Columns Ignored (Audit justification for excluded non-feature columns)
    Strictly handles missing values: if present, written with exact format; if missing, written as blank cell ("").
    Uses constant memory streaming for ultra-fast, zero-crash handling of 100k-500k+ enterprise rows.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else ".", exist_ok=True)

    column_mapping = [
        ('schema_name', 'Schema Source'),
        ('company_code', 'Company Code (BUKRS)'),
        ('company_name', 'Company Name (BUTXT)'),
        ('plant', 'Plant (WERKS)'),
        ('plant_name', 'Plant Name (NAME1)'),
        ('plant_city', 'Plant City (ORT01)'),
        ('company_currency', 'Currency (WAERS)'),
        ('material_id', 'Material Number (MATNR)'),
        ('material_desc', 'Material Description (MAKTX)'),
        ('po_short_text', 'PO Short Text (TXZ01)'),
        ('size_dimensions', 'Size / Dimensions (GROES)'),
        ('material_type', 'Material Type (MTART)'),
        ('material_type_desc', 'Material Type Description (MTBEZ)'),
        ('material_group', 'Current Material Group (MATKL)'),
        ('po_material_group', 'PO Material Group (EKPO-MATKL)'),
        ('auto_classified_matkl', 'NLP Recommended Group (MATKL)'),
        ('auto_classified_domain', 'NLP Auto-Classified Material Domain'),
        ('base_uom', 'Base UoM (MEINS)'),
        ('division', 'Division (SPART)'),
        ('purchasing_group', 'Purchasing Group (EKGRP)'),
        ('mrp_type', 'MRP Type (DISMM)'),
        ('procurement_type', 'Procurement Type (BESKZ)'),
        ('valuation_class', 'Valuation Class (BKLAS)'),
        ('price_control', 'Price Control (VPRSV)'),
        ('effective_unit_price', 'Unit Price (STPRS/VERPR)'),
        ('total_valuated_stock', 'Valuated Stock (LBKUM)'),
        ('total_stock_value', 'Total Stock Value (SALK3)'),
        ('po_line_count', 'PO Line Count (EKPO)'),
        ('total_po_spend', 'Total PO Spend (NETWR)'),
        ('created_date', 'Created Date (ERSDA)'),
        ('last_changed_date', 'Last Changed Date (LAEDA)'),
        ('desc_char_length', 'Desc Length (Chars)'),
        ('desc_word_count', 'Desc Word Count'),
        ('desc_spec_token_count', 'Spec Tokens Count'),
        ('tfidf_active_ngrams', 'Active TF-IDF N-Grams'),
        ('text_clarity_score', 'Text Clarity Score (0-100)'),
        ('semantic_confidence_score', 'NLP Semantic Confidence (0-100)'),
        ('taxonomy_alignment_score', 'Taxonomy Alignment Score (0-100)'),
        ('classification_tier_code', 'Classification Code (T-S-A)'),
        ('nlp_classification_confidence_score', 'NLP Classification Confidence'),
        ('nlp_misclassification_risk_index', 'Misclassification Risk Index'),
        ('material_cluster_id', 'ML Cluster ID'),
        ('material_segment_name', 'ML Material Classification Segment'),
        ('classification_exception_prob', 'Classification Exception Prob'),
        ('predicted_classification_exception', 'Predicted Classification Exception'),
        ('risk_category', 'Risk Category')
    ]

    export_cols = []
    display_headers = []
    for internal_col, display_name in column_mapping:
        if internal_col in df.columns:
            export_cols.append(internal_col)
            display_headers.append(display_name)

    df_export = df[export_cols].copy()

    # Format dates as DD-MM-YYYY or blank ""
    for dt_col in ['created_date', 'last_changed_date']:
        if dt_col in df_export.columns:
            dt_s = pd.to_datetime(df_export[dt_col], errors='coerce')
            df_export[dt_col] = dt_s.dt.strftime('%d-%m-%Y').fillna('')

    # Clean numeric rounding and clean string blanks
    for num_col in [
        'effective_unit_price', 'total_valuated_stock', 'total_stock_value', 'total_po_spend',
        'text_clarity_score', 'semantic_confidence_score', 'taxonomy_alignment_score',
        'nlp_classification_confidence_score', 'nlp_misclassification_risk_index'
    ]:
        if num_col in df_export.columns:
            df_export[num_col] = pd.to_numeric(df_export[num_col], errors='coerce').round(2)

    actual_path = output_path
    save_candidates = [
        output_path,
        "ML_24_Material_Auto_Classification_Enriched_Output_Clean.xlsx",
        "ML_24_Material_Auto_Classification_Enriched_Output_Final.xlsx",
        os.path.join("reports", os.path.basename(output_path)),
        f"ML_24_Material_Auto_Classification_Enriched_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    ]

    saved = False
    for candidate in save_candidates:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(candidate)) if os.path.dirname(candidate) else ".", exist_ok=True)
            wb = xlsxwriter.Workbook(candidate, {'constant_memory': True})
            wb.use_zip64()
            actual_path = candidate
            saved = True
            break
        except PermissionError:
            print(f"[WARNING] '{candidate}' is currently open in Excel / locked. Trying next file candidate...")
            continue

    if not saved:
        fallback = f"ML_24_Material_Classification_Output_{int(datetime.datetime.now().timestamp())}.xlsx"
        wb = xlsxwriter.Workbook(fallback, {'constant_memory': True})
        wb.use_zip64()
        actual_path = fallback

    # Format Styles
    meta_label_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#000000'})
    meta_val_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': False, 'font_color': '#000000'})
    header_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'bg_color': '#F2F2F2', 'border': 1, 'align': 'center', 'valign': 'vcenter'})

    cell_text_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1})
    cell_center_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'center'})
    cell_num_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'right', 'num_format': '#,##0.00'})
    cell_int_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'center', 'num_format': '#,##0'})
    cell_prob_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'right', 'num_format': '0.00000000'})

    crit_risk_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C0006', 'bg_color': '#FFC7CE', 'border': 1, 'align': 'center'})
    high_risk_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C6500', 'bg_color': '#FFEB9C', 'border': 1, 'align': 'center'})
    med_risk_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'font_color': '#006100', 'bg_color': '#C6EFCE', 'border': 1, 'align': 'center'})
    low_risk_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'center'})

    # -------------------------------------------------------------
    # Sheet 1: ML_24_Enriched_Output(Predict)
    # -------------------------------------------------------------
    ws1 = wb.add_worksheet("ML_24_Enriched_Output(Predict)")
    ws1.set_row(6, 24)

    gen_time_str = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    row_count_str = f"{len(df_export):,}"

    metadata_rows = [
        ("Insight ID", insight_id),
        ("Use Case", use_case),
        ("Description", description),
        ("Generated On", gen_time_str),
        ("Row Count", row_count_str)
    ]

    for r_idx, (lbl, val) in enumerate(metadata_rows):
        ws1.write(r_idx, 0, lbl, meta_label_fmt)
        ws1.write(r_idx, 1, val, meta_val_fmt)

    # Row 6 (index 5) is left blank intentionally as per enterprise SAP template

    # Row 7 (index 6): Headers
    for c_idx, header_text in enumerate(display_headers):
        ws1.write(6, c_idx, header_text, header_fmt)

    # Pre-determine column types for fast row-by-row cell formatting
    col_types = []
    for h in display_headers:
        if "Prob" in h:
            col_types.append("prob")
        elif any(k in h for k in ["Price", "Stock", "Value", "Spend", "Score", "Confidence", "Index"]):
            col_types.append("num")
        elif any(k in h for k in ["Count", "Length", "N-Grams", "Cluster ID", "Exception"]) and "Prob" not in h:
            col_types.append("int")
        elif h in [
            "Schema Source", "Company Code (BUKRS)", "Plant (WERKS)", "Currency (WAERS)",
            "Material Number (MATNR)", "Material Type (MTART)", "Current Material Group (MATKL)",
            "PO Material Group (EKPO-MATKL)", "NLP Recommended Group (MATKL)",
            "Base UoM (MEINS)", "Division (SPART)", "Purchasing Group (EKGRP)",
            "MRP Type (DISMM)", "Procurement Type (BESKZ)", "Valuation Class (BKLAS)",
            "Price Control (VPRSV)", "Classification Code (T-S-A)",
            "Created Date (ERSDA)", "Last Changed Date (LAEDA)"
        ]:
            col_types.append("center")
        elif h == "Risk Category":
            col_types.append("risk")
        else:
            col_types.append("text")

    # Data Rows (index 7 onwards)
    data_matrix = df_export.values
    for r_offset, row_vals in enumerate(data_matrix, start=7):
        for c_idx, val in enumerate(row_vals):
            ctype = col_types[c_idx]

            # Strict Missing Value Handling: if missing, write blank ""
            if val is None or pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                ws1.write_blank(r_offset, c_idx, None, cell_text_fmt)
                continue

            str_val = str(val).strip()
            if str_val in ('nan', 'None', 'NULL', '<NA>', 'NaT', ''):
                ws1.write_blank(r_offset, c_idx, None, cell_text_fmt)
                continue

            if ctype == "prob":
                try:
                    ws1.write_number(r_offset, c_idx, float(val), cell_prob_fmt)
                except (ValueError, TypeError):
                    ws1.write_string(r_offset, c_idx, str_val, cell_text_fmt)
            elif ctype == "num":
                try:
                    ws1.write_number(r_offset, c_idx, float(val), cell_num_fmt)
                except (ValueError, TypeError):
                    ws1.write_string(r_offset, c_idx, str_val, cell_text_fmt)
            elif ctype == "int":
                try:
                    ws1.write_number(r_offset, c_idx, int(round(float(val))), cell_int_fmt)
                except (ValueError, TypeError):
                    ws1.write_string(r_offset, c_idx, str_val, cell_text_fmt)
            elif ctype == "center":
                ws1.write_string(r_offset, c_idx, str_val, cell_center_fmt)
            elif ctype == "risk":
                if str_val == "Critical Risk":
                    ws1.write_string(r_offset, c_idx, str_val, crit_risk_fmt)
                elif str_val == "High Risk":
                    ws1.write_string(r_offset, c_idx, str_val, high_risk_fmt)
                elif str_val == "Medium Risk":
                    ws1.write_string(r_offset, c_idx, str_val, med_risk_fmt)
                else:
                    ws1.write_string(r_offset, c_idx, str_val, low_risk_fmt)
            else:
                ws1.write_string(r_offset, c_idx, str_val, cell_text_fmt)

    # -------------------------------------------------------------
    # Sheet 2: Feature_Importance
    # -------------------------------------------------------------
    if feature_importance_df is None and os.path.exists("reports/feature_importance.csv"):
        feature_importance_df = pd.read_csv("reports/feature_importance.csv")

    if feature_importance_df is not None and len(feature_importance_df) > 0:
        ws2 = wb.add_worksheet("Feature_Importance")
        ws2.write(0, 0, f"{insight_id} | Feature_Importance", meta_label_fmt)

        feat_hdr_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#FFFFFF', 'bg_color': '#2F5597', 'border': 1, 'valign': 'vcenter'})
        ws2.write(1, 0, "feature", feat_hdr_fmt)
        ws2.write(1, 1, "importance", feat_hdr_fmt)
        ws2.write(1, 2, "Direct Mathematical Influence On Material Auto Classification & Taxonomy:", feat_hdr_fmt)
        ws2.set_row(1, 22)

        feat_imp_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'right', 'num_format': '0.000000000'})
        col_f = 'feature' if 'feature' in feature_importance_df.columns else feature_importance_df.columns[0]
        col_i = 'importance' if 'importance' in feature_importance_df.columns else feature_importance_df.columns[1]
        has_e = len(feature_importance_df.columns) > 2
        col_e = feature_importance_df.columns[2] if has_e else None

        for idx, r_data in feature_importance_df.iterrows():
            curr_r = idx + 2
            ws2.write_string(curr_r, 0, str(r_data[col_f]), cell_text_fmt)
            ws2.write_number(curr_r, 1, float(r_data[col_i]), feat_imp_fmt)
            expl = str(r_data[col_e]) if col_e and pd.notna(r_data[col_e]) else ""
            ws2.write_string(curr_r, 2, expl, cell_text_fmt)

        ws2.set_column(0, 0, 34)
        ws2.set_column(1, 1, 18)
        ws2.set_column(2, 2, 88)

    # -------------------------------------------------------------
    # Sheet 3: Columns Ignored
    # -------------------------------------------------------------
    ws3 = wb.add_worksheet("Columns Ignored")
    ws3.write(0, 0, "Columns Ignored / Excluded from Model Training:", meta_label_fmt)

    ign_hdr_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#000000', 'border': 1, 'valign': 'vcenter'})
    ws3.write(2, 0, "Ignored Column", ign_hdr_fmt)
    ws3.write(2, 1, "SAP Field Name", ign_hdr_fmt)
    ws3.write(2, 2, "Why It Was Ignored", ign_hdr_fmt)
    ws3.set_row(2, 20)

    ignored_rows = [
        ("Material Number Primary Key", "MATNR", "Unique alphanumeric SAP Material Master primary key. Carries no intrinsic semantic signal on material description taxonomy."),
        ("Raw Unvectorized Description Strings", "MAKTX, MAKTG, TXZ01, GROES", "Raw text strings cannot be passed directly into Euclidean/Gaussian clustering algorithms; transformed into TF-IDF n-grams, 5 TruncatedSVD semantic components, and lexical domain affinities."),
        ("Old Material Number", "BISMT", "Legacy reference identifier retained strictly for master data traceability."),
        ("Plant & Storage Organizational Codes", "WERKS, NAME1, ORT01, REGIO", "Plant location metadata retained for operational drill-down rather than semantic description classification."),
        ("Created / Changed By User IDs", "ERNAM, AENAM", "SAP user identifiers who created or modified the material master record. Carry no semantic material taxonomy signal."),
        ("Raw Creation & Modification Dates", "ERSDA, LAEDA", "Calendar dates retained for audit lineage; material classification depends on description semantics and technical specifications."),
        ("Profit Center & Loading Group Codes", "PRCTR, LADGR, TRAGR", "Accounting and logistics handling codes retained for governance reporting."),
        ("Company Code & Currency", "BUKRS, BUTXT, WAERS", "Enterprise entity metadata constant across company code 1000."),
        ("Post-Hoc Benchmark Segment & Label", "benchmark_material_segment, is_classification_exception", "Strictly excluded from model training so all 5 NLP clustering models remain 100% unsupervised and classify materials purely from input feature distributions.")
    ]

    for r_offset, (col_name, sap_field, reason) in enumerate(ignored_rows, start=3):
        ws3.write_string(r_offset, 0, col_name, cell_text_fmt)
        ws3.write_string(r_offset, 1, sap_field, cell_text_fmt)
        ws3.write_string(r_offset, 2, reason, cell_text_fmt)

    ws3.set_column(0, 0, 32)
    ws3.set_column(1, 1, 28)
    ws3.set_column(2, 2, 98)

    wb.close()
    print(f"[SUCCESS] Executive Enriched Excel Report saved to: {actual_path}")
    print(f"[INFO] Total Rows: {len(df_export):,}, Total Columns: {len(display_headers)}, Sheets: ['ML_24_Enriched_Output(Predict)', 'Feature_Importance', 'Columns Ignored']")
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
    champ_path = os.path.join(output_dir, "material_classification_champion_model.joblib")
    bundle_path = os.path.join(output_dir, "material_classification_model_bundle.joblib")
    km_path = os.path.join(output_dir, "champion_kmeans.joblib")
    scaler_path = os.path.join(output_dir, "robust_scaler.joblib")

    joblib.dump(bundle, champ_path)
    joblib.dump(bundle, bundle_path)
    joblib.dump(model.model, km_path)
    joblib.dump(scaler, scaler_path)

    if suite is not None and 'gmm' in suite.models:
        joblib.dump(suite.models['gmm'], os.path.join(output_dir, "challenger_gmm.joblib"))

    print(f"[SUCCESS] Saved champion NLP clustering model bundle to: {champ_path}")
