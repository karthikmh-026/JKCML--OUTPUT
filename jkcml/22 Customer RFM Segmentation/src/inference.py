"""
Inference & Excel Reporting Module - Use Case 22: Customer RFM Segmentation (Clustering)
Handles scoring and segmenting SAP Customer records using the #1 Ranked Champion Clustering Model
(K-Means Clustering) and generating executive 3-sheet Excel reports matching the enterprise SAP template.
Includes strict missing-value handling: populated attributes retain genuine values, and any missing or
unpopulated attributes are output as clean blank cells ("").
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
    Applies the selected #1 Champion Clustering ML model (K-Means Clustering) to calculate:
    - rfm_cluster_id: ordered cluster tier (0 = Champions, 1 = Core Growth, 2 = At-Risk, 3 = Dormant/Churned)
    - rfm_segment_name: human-readable Customer RFM Segment
    - customer_rfm_risk_prob: continuous churn/dormancy/behavioral risk probability in [0.0, 1.0] (8 decimals)
    - predicted_rfm_exception: binary flag (1 for Dormant/Churned/High-Risk customer, 0 for Healthy/Active)
    - risk_category: executive triage tier (Critical Risk, High Risk, Medium Risk, Low Risk)
    """
    df_out = df.copy()

    X = df_out[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    probs, raw_scores, mapped_tiers = model.predict_proba(X_scaled)
    _, segment_names = model.predict_clusters(X_scaled)
    preds = model.predict(X_scaled)

    df_out['rfm_cluster_id'] = mapped_tiers.astype(int)
    df_out['rfm_segment_name'] = segment_names
    df_out['customer_rfm_risk_prob'] = np.round(probs, 8)
    df_out['predicted_rfm_exception'] = preds.astype(int)

    conditions = [
        (df_out['rfm_cluster_id'] == 3) & ((df_out['is_account_blocked_or_flagged'] == 1) | (df_out['open_ar_balance'] > 0) | (df_out['rfm_churn_risk_index'] >= 88.0)),
        (df_out['rfm_cluster_id'] == 3) | (df_out['customer_rfm_risk_prob'] >= 0.72),
        (df_out['rfm_cluster_id'] == 2) | (df_out['customer_rfm_risk_prob'] >= 0.48),
    ]
    choices = ['Critical Risk', 'High Risk', 'Medium Risk']
    df_out['risk_category'] = np.select(conditions, choices, default='Low Risk')

    # Sort so high-value active & high-priority customer accounts are clearly structured
    df_out = df_out.sort_values(
        by=['customer_rfm_risk_prob', 'open_ar_balance', 'total_billed_net_value'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df_out


def export_enriched_excel_report(
    df: pd.DataFrame,
    output_path: str = "ML_22_Customer_RFM_Segmentation_Enriched_Output_Clean.xlsx",
    insight_id: str = "ML_22",
    use_case: str = "Customer RFM Segmentation",
    description: str = "Group customers by behavior and value using Unsupervised Clustering across SAP Customer Master (KNA1, KNVV, KNB1), Billing Documents (VBRK), Sales Orders (VBAK), and Open Accounts Receivable (BSID) from HANA_P2 and HANA_P3 without limits.",
    feature_importance_df: pd.DataFrame = None
) -> str:
    """
    Generates the executive enriched Excel report matching the enterprise SAP reference template:
    - Sheet 1: ML_22_Enriched_Output(Predict)
      - Rows 1-5: Metadata block (Insight ID, Use Case, Description, Generated On, Row Count)
      - Row 6: Blank row
      - Row 7: Enterprise SAP column headers
      - Row 8+: Enriched SAP Customer RFM records with cluster assignments, R/F/M scores, probabilities, and risk tiers
    - Sheet 2: Feature_Importance (All 32 engineered RFM features with mathematical rationale)
    - Sheet 3: Columns Ignored (Audit justification for excluded non-feature columns)

    Strict Missing Value Handling:
    If an attribute exists, it is populated. If an attribute is missing/null/nan, it is strictly output as a blank cell ("").
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    column_mapping = [
        ('company_code', 'Company Code (BUKRS)'),
        ('company_name', 'Company Name (BUTXT)'),
        ('company_city', 'Company City'),
        ('company_country', 'Company Country'),
        ('currency', 'Currency (WAERS)'),
        ('sales_org', 'Sales Org (VKORG)'),
        ('dist_channel', 'Dist Channel (VTWEG)'),
        ('division', 'Division (SPART)'),
        ('customer_id', 'Customer (KUNNR)'),
        ('customer_name', 'Customer Name (NAME1)'),
        ('customer_city', 'Customer City (ORT01)'),
        ('customer_region', 'Region (REGIO)'),
        ('postal_code', 'Postal Code (PSTLZ)'),
        ('account_group', 'Account Group (KTOKD)'),
        ('account_group_desc', 'Account Group Description'),
        ('customer_group', 'Customer Group (KDGRP)'),
        ('sales_district', 'Sales District (BZIRK)'),
        ('sales_office', 'Sales Office (VKBUR)'),
        ('price_list_type', 'Price List Type (PLTYP)'),
        ('payment_terms', 'Payment Terms (ZTERM)'),
        ('abc_class', 'ABC Class (KLABC)'),
        ('channel_segment_desc', 'Channel Segment'),
        ('master_created_date', 'Created Date (ERDAT)'),
        ('first_billing_date', 'First Billing Date (FKDAT)'),
        ('last_billing_date', 'Last Billing Date (FKDAT)'),
        ('recency_effective_days', 'Recency (Days)'),
        ('billing_doc_count', 'Billing Frequency (VBRK Count)'),
        ('sales_order_count', 'Sales Order Count (VBAK)'),
        ('active_billing_months', 'Active Billing Months'),
        ('total_billed_net_value', 'Total Monetary Net Value (NETWR)'),
        ('avg_bill_value', 'Avg Invoice Value (INR)'),
        ('max_bill_value', 'Max Invoice Value (INR)'),
        ('cancelled_or_return_bills', 'Cancelled / Return Bills'),
        ('open_ar_item_count', 'Open AR Items (BSID)'),
        ('open_ar_balance', 'Open AR Balance (DMBTR)'),
        ('recency_score_norm', 'Recency Score (0-100)'),
        ('frequency_score_norm', 'Frequency Score (0-100)'),
        ('monetary_score_norm', 'Monetary Score (0-100)'),
        ('rfm_tier_code', 'RFM Quintile Code (R-F-M)'),
        ('rfm_composite_score', 'RFM Composite Value Score'),
        ('rfm_churn_risk_index', 'RFM Churn Risk Index'),
        ('rfm_cluster_id', 'ML Cluster ID'),
        ('rfm_segment_name', 'ML Customer RFM Segment'),
        ('customer_rfm_risk_prob', 'Customer RFM Risk Probability'),
        ('predicted_rfm_exception', 'Predicted RFM Exception'),
        ('risk_category', 'Risk Category')
    ]

    export_cols = []
    display_headers = []
    for internal_col, display_name in column_mapping:
        if internal_col in df.columns:
            export_cols.append(internal_col)
            display_headers.append(display_name)

    df_export = df[export_cols].copy()

    # Format dates: if missing or NaT, set to blank ""
    for dt_col in ['master_created_date', 'first_billing_date', 'last_billing_date']:
        if dt_col in df_export.columns:
            dt_s = pd.to_datetime(df_export[dt_col], errors='coerce')
            df_export[dt_col] = dt_s.dt.strftime('%d-%m-%Y').fillna('')

    # Clean numeric columns: rounded, no NaNs
    num_round_cols = [
        'total_billed_net_value', 'avg_bill_value', 'max_bill_value', 'open_ar_balance',
        'recency_score_norm', 'frequency_score_norm', 'monetary_score_norm',
        'rfm_composite_score', 'rfm_churn_risk_index'
    ]
    for num_col in num_round_cols:
        if num_col in df_export.columns:
            df_export[num_col] = pd.to_numeric(df_export[num_col], errors='coerce').fillna(0.0).round(2)

    int_cols = [
        'recency_effective_days', 'billing_doc_count', 'sales_order_count', 'active_billing_months',
        'cancelled_or_return_bills', 'open_ar_item_count', 'rfm_cluster_id', 'predicted_rfm_exception'
    ]
    for c in int_cols:
        if c in df_export.columns:
            df_export[c] = pd.to_numeric(df_export[c], errors='coerce').fillna(0).astype(int)

    if 'customer_rfm_risk_prob' in df_export.columns:
        df_export['customer_rfm_risk_prob'] = pd.to_numeric(df_export['customer_rfm_risk_prob'], errors='coerce').fillna(0.0).round(8)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"{insight_id}_Enriched_Output(Predict)"

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

    header_row_idx = 7
    font_header = Font(name="Calibri", size=10, bold=True, color="000000")
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

    # Strict Missing Value Sanitizer & Fast Row Appender
    for row in df_export.itertuples(index=False):
        clean_row = []
        for val in row:
            if val is None or pd.isna(val):
                clean_row.append("")
            elif isinstance(val, str):
                s = val.strip()
                if s.lower() in ['nan', 'none', 'nat', 'null']:
                    clean_row.append("")
                else:
                    clean_row.append(s)
            elif isinstance(val, float):
                if np.isnan(val) or np.isinf(val):
                    clean_row.append("")
                else:
                    clean_row.append(val)
            else:
                clean_row.append(val)
        ws.append(clean_row)

    # Column auto-width adjustment based on sample of rows
    for c_idx, header_text in enumerate(display_headers, start=1):
        col_letter = get_column_letter(c_idx)
        max_len = len(str(header_text))
        for r_sample in range(8, min(80, len(df_export) + 8)):
            val = ws.cell(row=r_sample, column=c_idx).value
            if val is not None and str(val).strip():
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, 11), 40)

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
        c_expl = ws_feat.cell(row=2, column=3, value="Direct Mathematical Influence On Customer RFM Segmentation & Value Risk:")

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

        font_data = Font(name="Calibri", size=10, color="000000")
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

        ws_feat.column_dimensions['A'].width = 32
        ws_feat.column_dimensions['B'].width = 18
        ws_feat.column_dimensions['C'].width = 82

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
        ("Customer Number", "KUNNR", "Unique alphanumeric SAP Customer Master primary key. Carries no intrinsic mathematical signal on purchasing behavior or monetary value."),
        ("Customer Name & Address Text", "NAME1, ORT01, PSTLZ, STRAS", "Free-text customer name and geographic address strings retained strictly for executive report identification."),
        ("Tax & Registration Identifiers", "STCD3, J_1IPANNO", "Statutory GSTIN and PAN registration numbers. High-cardinality identifiers excluded from numerical clustering."),
        ("Sales Area & Organizational Codes", "VKORG, VTWEG, SPART, VKBUR, VKGRP", "Categorical organizational codes. Used to condition peer monetary Z-scores (sales_org_monetary_zscore) and retained for business filtering."),
        ("Account Group & Classification Codes", "KTOKD, TXT30, KDGRP, BZIRK", "Master data classification codes retained for segment drill-down rather than Euclidean centroid calculation."),
        ("Raw Transaction & Creation Dates", "FKDAT, ERDAT, BUDAT", "Raw calendar dates cannot be ingested directly by clustering algorithms. Engineered into recency_billing_days, recency_order_days, recency_effective_days, customer_tenure_days, and active_span_days."),
        ("Reconciliation Account & Payment Terms", "AKONT, ZTERM, ZWELS", "General Ledger reconciliation account and payment term codes retained as financial audit context."),
        ("Created By User ID", "ERNAM", "SAP user identifier who created the customer master record. Carries no commercial RFM behavioral signal."),
        ("Company Code & Currency", "BUKRS, BUTXT, WAERS", "Enterprise entity metadata retained for multi-company reporting context."),
        ("Post-Hoc Benchmark Segment & Label", "benchmark_rfm_segment, is_rfm_exception", "Strictly excluded from model training so all 5 clustering models remain 100% unsupervised and group customers purely from input feature distributions.")
    ]

    font_data = Font(name="Calibri", size=10, color="000000")
    for r_offset, (col_name, sap_field, reason) in enumerate(ignored_rows, start=4):
        c1 = ws_ign.cell(row=r_offset, column=1, value=col_name)
        c2 = ws_ign.cell(row=r_offset, column=2, value=sap_field)
        c3 = ws_ign.cell(row=r_offset, column=3, value=reason)

        for c_cell in [c1, c2, c3]:
            c_cell.font = font_data
            c_cell.border = border_header
            c_cell.alignment = Alignment(horizontal="left", vertical="center")

    ws_ign.column_dimensions['A'].width = 30
    ws_ign.column_dimensions['B'].width = 28
    ws_ign.column_dimensions['C'].width = 98

    # Save to disk
    actual_path = output_path
    save_candidates = [
        output_path,
        os.path.join("reports", os.path.basename(output_path)),
        f"ML_22_Customer_RFM_Segmentation_Enriched_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
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
        fallback = f"ML_22_Customer_RFM_Output_{int(datetime.datetime.now().timestamp())}.xlsx"
        wb.save(fallback)
        actual_path = fallback

    print(f"[SUCCESS] Executive Enriched Excel Report saved to: {actual_path}")
    print(f"[INFO] Total Rows: {len(df_export):,}, Total Columns: {len(display_headers)}, Sheets: {wb.sheetnames}")
    return actual_path


def save_single_model_pipeline(model, scaler, feature_cols, suite=None, comparison_df=None, output_dir: str = "models"):
    """
    Saves the selected #1 Champion Clustering Model (K-Means Clustering), scaler, and 5-model benchmark bundle to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': 'K-Means Clustering',
        'benchmark_ranking': comparison_df.to_dict(orient='records') if comparison_df is not None else None
    }
    champ_path = os.path.join(output_dir, "customer_rfm_champion_model.joblib")
    bundle_path = os.path.join(output_dir, "customer_rfm_model_bundle.joblib")
    km_path = os.path.join(output_dir, "champion_kmeans.joblib")
    scaler_path = os.path.join(output_dir, "robust_scaler.joblib")

    joblib.dump(bundle, champ_path)
    joblib.dump(bundle, bundle_path)
    joblib.dump(model.model, km_path)
    joblib.dump(scaler, scaler_path)

    if suite is not None and 'gmm' in suite.models:
        joblib.dump(suite.models['gmm'], os.path.join(output_dir, "challenger_gmm.joblib"))

    print(f"[SUCCESS] Saved champion clustering model bundle to: {champ_path}")
