"""
Main Execution Script - Use Case 23: Vendor Risk Segmentation (Clustering)
Runs the end-to-end Unsupervised Machine Learning Clustering pipeline:
1. Connects to PostgreSQL server (192.168.1.44:5432 / JKC_Input / HANA_P2) and extracts SAP Vendor Master,
   Purchase Orders, Invoice Receipts, Blocked Invoices, Open Accounts Payable, and Quality Inspection Lot records
   from LFA1, LFB1, LFBK, EKKO, EKPO, RBKP, RBKP_BLOCKED, BSIK, QALS, T077Y, T001.
2. Performs data cleaning and engineers 32 domain-specific features for Vendor Risk Segmentation (Reliability & Risk).
3. Trains 5 Unsupervised Clustering ML models (K-Means Clustering, Gaussian Mixture Model, Mini-Batch K-Means,
   Bisecting K-Means, BIRCH Clustering) strictly on input features X (no output labels used during training).
4. Checks Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Silhouette Score, and Specificity to rank all 5 models
   and selects the #1 Champion Model based on the ranking.
5. Uses the selected #1 Champion Model (K-Means Clustering) to segment all extracted vendor records into
   4 reliability & risk clusters (Strategic, Preferred, Watchlist, High-Risk/Non-Compliant) and scores them with
   calibrated vendor risk probabilities, binary predictions, and risk categories (Critical, High, Medium, Low Risk).
6. Computes 32-feature importance and exports the executive 3-sheet Excel reports and visual evaluation charts.
"""
import os
import shutil
import argparse
import pandas as pd
from src.db_connector import load_config, extract_vendor_risk_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import VendorRiskClusteringSuite, SingleModelVendorRiskSegmenter
from src.evaluate import (
    evaluate_models,
    plot_model_comparison,
    plot_roc_pr_curves,
    compute_feature_importance
)
from src.inference import (
    score_with_single_model,
    export_enriched_excel_report,
    save_single_model_pipeline
)


def run_pipeline(
    limit: int = None,
    n_clusters: int = 4,
    output_excel: str = "ML_23_Vendor_Risk_Segmentation_Enriched_Output_Clean.xlsx"
):
    print("=" * 96)
    print("  USE CASE 23: VENDOR RISK SEGMENTATION (CLUSTERING) - 5 UNSUPERVISED ML MODELS -> #1 CHAMPION")
    print("=" * 96)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    db_cfg = config.get("database", {})
    schemas_list = db_cfg.get("schemas", [db_cfg.get("schema", "HANA_P2")])
    schemas_str = " & ".join(schemas_list)
    print(f"\n>>> STEP 1: Connecting to PostgreSQL Server ({db_cfg.get('host')}:{db_cfg.get('port')} / Schemas: {schemas_str}) & Extracting SAP Vendors...")
    print("    Extracting joined data across: LFA1, LFB1, LFBK, EKKO, EKPO, RBKP, RBKP_BLOCKED, BSIK, QALS, T077Y, T001")
    raw_df = extract_vendor_risk_data(limit=limit, config=config)

    # 2. Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing Data Cleaning & Engineering 32 Vendor Reliability & Risk Features...")
    featured_df = clean_and_engineer_features(raw_df)

    exception_cnt = int(featured_df['is_vendor_risk_exception'].sum())
    exception_rate = (exception_cnt / len(featured_df)) * 100.0
    contamination = max(0.05, min(0.45, exception_cnt / max(1, len(featured_df))))
    print(f"[SUMMARY] Total Extracted SAP Vendor Records: {len(featured_df):,}")
    print(f"[SUMMARY] Benchmark High-Risk / Non-Compliant Vendor Exceptions: {exception_cnt:,} ({exception_rate:.2f}%)")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 32-Feature Vendor Risk Matrix & Applying RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train & Compare 5 Unsupervised Clustering Models
    print("\n>>> STEP 4: Training 5 Unsupervised Clustering ML Models (Input-Only Training, No Output Labels)...")
    suite = VendorRiskClusteringSuite(n_clusters=n_clusters, contamination=contamination, random_state=42)
    suite.train_all(X_train_scaled)

    print("\n>>> STEP 5: Evaluating Accuracy & Ranking the 5 Unsupervised Clustering ML Models on Test Split...")
    test_preds = suite.predict_all(X_test_scaled, y_benchmark=y_test)
    comparison_df = evaluate_models(test_preds, y_test)

    print("\n" + "=" * 96)
    print("  5 UNSUPERVISED CLUSTERING ML MODELS RANKING & ACCURACY TABLE (TEST SET EVALUATION)")
    print("=" * 96)
    print(comparison_df.to_string(index=False))
    print("=" * 96)

    os.makedirs("reports/figures", exist_ok=True)
    comparison_df.to_csv("reports/model_comparison_metrics.csv", index=False)
    plot_model_comparison(comparison_df, output_dir="reports/figures")
    plot_roc_pr_curves(test_preds, y_test, output_dir="reports/figures")

    champion_model_name = comparison_df.iloc[0]['Model']
    champion_acc = comparison_df.iloc[0]['Accuracy'] * 100.0
    champion_prec = comparison_df.iloc[0]['Precision'] * 100.0
    champion_f1 = comparison_df.iloc[0]['F1-Score']
    champion_auc = comparison_df.iloc[0]['ROC-AUC']
    champion_sil = comparison_df.iloc[0]['Silhouette']
    print(f"\n[CHAMPION SELECTED] Rank #1 Model: {champion_model_name}")
    print(f"  -> Accuracy: {champion_acc:.2f}% | Precision: {champion_prec:.2f}% | F1-Score: {champion_f1:.4f} | ROC-AUC: {champion_auc:.4f} | Silhouette: {champion_sil:.4f}")

    # 6. Fit Selected #1 Champion Model (K-Means Clustering) & Segment All Vendor Records
    print(f"\n>>> STEP 6: Segmenting & Scoring All {len(featured_df):,} SAP Vendors with #1 Champion Model ({champion_model_name})...")
    segmenter = SingleModelVendorRiskSegmenter(
        n_clusters=n_clusters,
        contamination=contamination,
        random_state=42,
        pretrained_model=suite.models['kmeans']
    )
    segmenter.fit(X_train_scaled)
    scored_df = score_with_single_model(featured_df, segmenter, scaler, feature_cols)

    print("\n[VENDOR RISK CLUSTERING SEGMENT DISTRIBUTION]:")
    for seg_name, cnt in scored_df['vendor_segment_name'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        print(f"  - {seg_name:<44}: {cnt:>6,} vendors ({pct:.2f}%)")

    crit_cnt = int((scored_df['risk_category'] == 'Critical Risk').sum())
    high_cnt = int((scored_df['risk_category'] == 'High Risk').sum())
    med_cnt = int((scored_df['risk_category'] == 'Medium Risk').sum())
    low_cnt = int((scored_df['risk_category'] == 'Low Risk').sum())

    print("\n[EXECUTIVE VENDOR RISK TRIAGE SUMMARY]:")
    print(f"  - Critical Risk: {crit_cnt:,} vendors (high-risk accounts with blocked invoices, master blocks, or open AP exposure)")
    print(f"  - High Risk:     {high_cnt:,} vendors (non-compliant/low-reliability suppliers requiring procurement audit)")
    print(f"  - Medium Risk:   {med_cnt:,} vendors (watchlist suppliers with moderate compliance or delivery friction)")
    print(f"  - Low Risk:      {low_cnt:,} vendors (strategic & preferred suppliers with strong delivery and invoice compliance)")

    # Save top high-risk vendor watchlist CSV
    scored_df[scored_df['risk_category'].isin(['Critical Risk', 'High Risk', 'Medium Risk'])].head(1500).to_csv(
        "reports/vendor_risk_segmentation_watchlist.csv", index=False
    )

    # 7. Compute 32-Feature Importance & Explainability
    print("\n>>> STEP 7: Computing 32-Feature Importance for Vendor Risk Segmentation...")
    feat_imp = compute_feature_importance(segmenter.model, feature_cols, X_train_scaled, y_true=y_train)
    print("\n=== TOP 15 FEATURE IMPORTANCE SCORES ===")
    print(feat_imp.head(15).to_string(index=False))
    feat_imp.to_csv("reports/feature_importance.csv", index=False)

    # 8. Export Enriched 3-Sheet Excel Reports
    print("\n>>> STEP 8: Generating Executive 3-Sheet Enriched Excel Report...")
    desc_text = (
        "Group vendors by reliability and risk using Unsupervised Clustering across SAP Vendor Master "
        "(LFA1, LFB1, LFBK), Purchase Orders (EKKO, EKPO), Invoice Receipts & Blocked Invoices (RBKP, RBKP_BLOCKED), "
        "Open Accounts Payable (BSIK), Quality Inspection Lots (QALS), Account Groups (T077Y), and Company Code (T001)."
    )
    excel_path = export_enriched_excel_report(
        df=scored_df,
        output_path=output_excel,
        insight_id="ML_23",
        use_case="Vendor Risk Segmentation",
        description=desc_text,
        feature_importance_df=feat_imp
    )

    # Also save standard named files in root and reports/
    for extra_path in [
        "ML_23_Vendor_Risk_Segmentation_Enriched_Output.xlsx",
        os.path.join("reports", "ML_23_Vendor_Risk_Segmentation_Enriched_Output.xlsx"),
        os.path.join("reports", os.path.basename(output_excel))
    ]:
        if os.path.abspath(extra_path) != os.path.abspath(excel_path):
            os.makedirs(os.path.dirname(os.path.abspath(extra_path)), exist_ok=True)
            if os.path.exists(excel_path):
                try:
                    shutil.copyfile(excel_path, extra_path)
                    print(f"[INFO] Copied executive Excel report to: {extra_path}")
                except PermissionError:
                    print(f"[WARNING] Could not overwrite '{extra_path}' because it is currently open in Excel. Please close it if you wish to overwrite this specific file.")
                except Exception as e:
                    print(f"[WARNING] Could not copy to '{extra_path}': {e}")
            else:
                try:
                    export_enriched_excel_report(
                        df=scored_df,
                        output_path=extra_path,
                        insight_id="ML_23",
                        use_case="Vendor Risk Segmentation",
                        description=desc_text,
                        feature_importance_df=feat_imp
                    )
                except PermissionError:
                    print(f"[WARNING] Could not save to '{extra_path}' because it is currently open in Excel.")

    # 9. Persist Model Artifacts
    print("\n>>> STEP 9: Persisting Model Pipeline & Benchmark Artifacts...")
    save_single_model_pipeline(
        segmenter, scaler, feature_cols, suite=suite, comparison_df=comparison_df, output_dir="models"
    )

    print("\n" + "=" * 96)
    print("  VENDOR RISK SEGMENTATION CLUSTERING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  CHAMPION MODEL: {champion_model_name} (Accuracy: {champion_acc:.2f}%, Precision: {champion_prec:.2f}%)")
    print(f"  EXCEL REPORT GENERATED: {os.path.abspath(excel_path)}")
    print("=" * 96)
    return excel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAP Vendor Risk Segmentation with 5 Unsupervised Clustering ML Models & Champion Excel Output")
    parser.add_argument("--limit", type=int, default=None, help="Number of vendor records to extract (default: None - all records across HANA_P2 and HANA_P3)")
    parser.add_argument("--clusters", type=int, default=4, help="Number of vendor risk clusters (default: 4)")
    parser.add_argument("--output", type=str, default="ML_23_Vendor_Risk_Segmentation_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, n_clusters=args.clusters, output_excel=args.output)
