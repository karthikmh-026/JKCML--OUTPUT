"""
Main Execution Script - Use Case 22: Customer RFM Segmentation (Clustering)
Runs the end-to-end Unsupervised Machine Learning Clustering pipeline without limits:
1. Connects to PostgreSQL server (192.168.1.44:5432 / JKC_Input / HANA_P2 & HANA_P3) and extracts all SAP Customer
   Master, Billing Documents, Sales Orders, and Open Accounts Receivable records from KNA1, KNVV, KNB1, VBRK, VBAK, BSID, T077X, T001.
2. Performs data cleaning and engineers 32 domain-specific features for Customer RFM Segmentation (Recency, Frequency, Monetary Value).
   Ensures strict missing value handling: populated fields retain values, and missing fields are set to clean blanks ("").
3. Trains 5 Unsupervised Clustering ML models (K-Means Clustering, Gaussian Mixture Model, Mini-Batch K-Means,
   Bisecting K-Means, BIRCH Clustering) strictly on input features X (no output labels used during training).
4. Checks Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Silhouette Score, and Specificity to rank all 5 models
   and selects the #1 Champion Model based on the ranking.
5. Uses the selected #1 Champion Model (K-Means Clustering) to segment all extracted customer records into
   4 behavioral RFM value clusters (Champions, Core Growth, At-Risk, Dormant/Churned) and scores them with
   calibrated RFM risk probabilities, binary predictions, and risk categories (Critical, High, Medium, Low Risk).
6. Computes 32-feature importance and exports the executive 3-sheet Excel reports and visual evaluation charts.
"""
import os
import argparse
import pandas as pd
from src.db_connector import load_config, extract_customer_rfm_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import RFMClusteringSuite, SingleModelCustomerRFMSegmenter
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
    output_excel: str = "ML_22_Customer_RFM_Segmentation_Enriched_Output_Clean.xlsx"
):
    print("=" * 96)
    print("  USE CASE 22: CUSTOMER RFM SEGMENTATION (CLUSTERING) - ALL DATA (HANA_P2 & HANA_P3) - NO LIMIT")
    print("=" * 96)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    limit_text = f"LIMIT {limit}" if limit and int(limit) > 0 else "NO LIMIT (ALL DATA FROM SERVER)"
    print(f"\n>>> STEP 1: Connecting to PostgreSQL Server (192.168.1.44:5432 / HANA_P2 & HANA_P3) [{limit_text}]...")
    print("    Extracting full joined data from: KNA1, KNVV, KNB1, VBRK, VBAK, BSID, T077X, T001")
    raw_df = extract_customer_rfm_data(limit=limit, config=config)

    # 2. Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing Data Cleaning & Engineering 32 Customer RFM Behavioral Features...")
    print("    Handling missing values: all populated attributes preserved, missing values sanitized to clean blank cells.")
    featured_df = clean_and_engineer_features(raw_df)

    exception_cnt = int(featured_df['is_rfm_exception'].sum())
    exception_rate = (exception_cnt / len(featured_df)) * 100.0
    contamination = max(0.05, min(0.45, exception_cnt / max(1, len(featured_df))))
    print(f"[SUMMARY] Total Extracted SAP Customer Records: {len(featured_df):,}")
    print(f"[SUMMARY] Benchmark High-Churn / Dormant RFM Exceptions: {exception_cnt:,} ({exception_rate:.2f}%)")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 32-Feature RFM Matrix & Applying RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train & Compare 5 Unsupervised Clustering Models
    print("\n>>> STEP 4: Training 5 Unsupervised Clustering ML Models (Input-Only Training, No Output Labels)...")
    suite = RFMClusteringSuite(n_clusters=n_clusters, contamination=contamination, random_state=42)
    suite.train_all(X_train_scaled)

    print("\n>>> STEP 5: Evaluating Accuracy & Ranking the 5 Unsupervised Clustering ML Models on Test Split...")
    test_preds = suite.predict_all(X_test_scaled)
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

    # 6. Fit Selected #1 Champion Model (K-Means Clustering) & Segment All Customer Records
    print(f"\n>>> STEP 6: Segmenting & Scoring All {len(featured_df):,} SAP Customers with #1 Champion Model ({champion_model_name})...")
    segmenter = SingleModelCustomerRFMSegmenter(
        n_clusters=n_clusters,
        contamination=contamination,
        random_state=42,
        pretrained_model=suite.models['kmeans']
    )
    segmenter.fit(X_train_scaled)
    scored_df = score_with_single_model(featured_df, segmenter, scaler, feature_cols)

    print("\n[CUSTOMER RFM CLUSTERING SEGMENT DISTRIBUTION]:")
    for seg_name, cnt in scored_df['rfm_segment_name'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        print(f"  - {seg_name:<34}: {cnt:>7,} customers ({pct:.2f}%)")

    crit_cnt = int((scored_df['risk_category'] == 'Critical Risk').sum())
    high_cnt = int((scored_df['risk_category'] == 'High Risk').sum())
    med_cnt = int((scored_df['risk_category'] == 'Medium Risk').sum())
    low_cnt = int((scored_df['risk_category'] == 'Low Risk').sum())

    print("\n[EXECUTIVE CUSTOMER RFM RISK TRIAGE SUMMARY]:")
    print(f"  - Critical Risk: {crit_cnt:,} customers (dormant/churned accounts with open AR or blocks - immediate retention/credit action)")
    print(f"  - High Risk:     {high_cnt:,} customers (dormant/churned low-RFM accounts requiring reactivation campaigns)")
    print(f"  - Medium Risk:   {med_cnt:,} customers (at-risk/cooling accounts showing declining purchase frequency)")
    print(f"  - Low Risk:      {low_cnt:,} customers (champions & core growth accounts with strong RFM value)")

    # Save top high-risk customer watchlist CSV
    scored_df[scored_df['risk_category'].isin(['Critical Risk', 'High Risk', 'Medium Risk'])].head(1500).to_csv(
        "reports/customer_rfm_segmentation_watchlist.csv", index=False
    )

    # 7. Compute 32-Feature Importance & Explainability
    print("\n>>> STEP 7: Computing 32-Feature Importance for Customer RFM Segmentation...")
    feat_imp = compute_feature_importance(segmenter.model, feature_cols, X_train_scaled, y_true=y_train)
    print("\n=== TOP 15 FEATURE IMPORTANCE SCORES ===")
    print(feat_imp.head(15).to_string(index=False))
    feat_imp.to_csv("reports/feature_importance.csv", index=False)

    # 8. Export Enriched 3-Sheet Excel Reports
    print("\n>>> STEP 8: Generating Executive 3-Sheet Enriched Excel Report...")
    desc_text = (
        "Group customers by behavior and value using Unsupervised Clustering across SAP Customer Master "
        "(KNA1, KNVV, KNB1), Billing Documents (VBRK), Sales Orders (VBAK), Open Accounts Receivable (BSID), "
        "Account Groups (T077X), and Company Code (T001) from HANA_P2 and HANA_P3 without limits."
    )
    excel_path = export_enriched_excel_report(
        df=scored_df,
        output_path=output_excel,
        insight_id="ML_22",
        use_case="Customer RFM Segmentation",
        description=desc_text,
        feature_importance_df=feat_imp
    )

    # Also save standard named files in root and reports/ (for both ML_22 and ML_20 references)
    for extra_path, insight_tag in [
        ("ML_22_Customer_RFM_Segmentation_Enriched_Output.xlsx", "ML_22"),
        (os.path.join("reports", "ML_22_Customer_RFM_Segmentation_Enriched_Output.xlsx"), "ML_22"),
        (os.path.join("reports", "ML_22_Customer_RFM_Segmentation_Enriched_Output_Clean.xlsx"), "ML_22"),
        ("ML_20_Customer_RFM_Segmentation_Enriched_Output_Clean.xlsx", "ML_20"),
        ("ML_20_Customer_RFM_Segmentation_Enriched_Output.xlsx", "ML_20"),
        (os.path.join("reports", "ML_20_Customer_RFM_Segmentation_Enriched_Output.xlsx"), "ML_20"),
        (os.path.join("reports", "ML_20_Customer_RFM_Segmentation_Enriched_Output_Clean.xlsx"), "ML_20"),
    ]:
        if os.path.abspath(extra_path) != os.path.abspath(excel_path):
            export_enriched_excel_report(
                df=scored_df,
                output_path=extra_path,
                insight_id=insight_tag,
                use_case="Customer RFM Segmentation",
                description=desc_text,
                feature_importance_df=feat_imp
            )

    # 9. Persist Model Artifacts
    print("\n>>> STEP 9: Persisting Model Pipeline & Benchmark Artifacts...")
    save_single_model_pipeline(
        segmenter, scaler, feature_cols, suite=suite, comparison_df=comparison_df, output_dir="models"
    )

    print("\n" + "=" * 96)
    print("  CUSTOMER RFM SEGMENTATION CLUSTERING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  CHAMPION MODEL: {champion_model_name} (Accuracy: {champion_acc:.2f}%, Precision: {champion_prec:.2f}%)")
    print(f"  EXCEL REPORT GENERATED: {os.path.abspath(excel_path)}")
    print("=" * 96)
    return excel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAP Customer RFM Segmentation with 5 Unsupervised Clustering ML Models & Champion Excel Output")
    parser.add_argument("--limit", type=int, default=0, help="Number of customer records to extract (0 or omit for NO LIMIT / all data)")
    parser.add_argument("--clusters", type=int, default=4, help="Number of RFM clusters (default: 4)")
    parser.add_argument("--output", type=str, default="ML_22_Customer_RFM_Segmentation_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    actual_limit = args.limit if args.limit and args.limit > 0 else None
    run_pipeline(limit=actual_limit, n_clusters=args.clusters, output_excel=args.output)
