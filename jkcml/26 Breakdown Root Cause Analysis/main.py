"""
Main Execution Script - Use Case 26: Breakdown Root Cause Analysis (NLP)
Runs the end-to-end Unsupervised NLP & Clustering Machine Learning pipeline:
1. Connects to PostgreSQL server (192.168.1.44:5432 / JKC_Input / HANA_P2) and extracts SAP Plant
   Maintenance (PM) Orders, Breakdown & Maintenance Texts, Technician Confirmation Texts, Consumed
   Spare Parts & Material Descriptions, Component Reservations, and Work Center Metadata from
   AUFK, AFKO, AFRU, AUFM, MAKT, RESB, CRHD, T001W, and T001.
2. Performs NLP data cleaning, TF-IDF + TruncatedSVD semantic vectorization, lexical failure-mode
   root-cause affinity scoring, and engineers 32 domain-specific features to extract root causes from
   maintenance text.
3. Trains 5 Unsupervised NLP Clustering ML models (K-Means Clustering, Gaussian Mixture Model, Mini-Batch K-Means,
   Bisecting K-Means, BIRCH Clustering) strictly on input features X (no output labels used during training).
4. Checks Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Silhouette Score, and Specificity to rank all 5 models
   and selects the #1 Champion Model based on the ranking.
5. Uses the selected #1 Champion Model (K-Means Clustering) to extract root causes and segment all maintenance
   orders into 4 Breakdown & RCA tiers and scores them with calibrated breakdown exception probabilities,
   binary predictions, and risk categories (Critical, High, Medium, Low Risk).
6. Computes 32-feature importance and exports the executive 3-sheet Excel reports and visual evaluation charts.
"""
import os
import argparse
import pandas as pd
from src.db_connector import load_config, extract_breakdown_rca_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import BreakdownRCAClusteringSuite, SingleModelBreakdownRCAAnalyzer
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
    output_excel: str = "ML_26_Breakdown_Root_Cause_Analysis_Enriched_Output_Clean.xlsx"
):
    print("=" * 102)
    print("  USE CASE 26: BREAKDOWN ROOT CAUSE ANALYSIS (NLP) - 5 UNSUPERVISED ML MODELS -> #1 CHAMPION")
    print("=" * 102)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    print("\n>>> STEP 1: Connecting to PostgreSQL Server (192.168.1.44:5432 / HANA_P2 & HANA_P3) & Extracting ALL SAP PM Orders...")
    print("    Extracting joined data from: AUFK, AFKO, AFRU, AUFM, MAKT, CRHD, T001W, T001 without limit")
    raw_df = extract_breakdown_rca_data(limit=limit, config=config)

    # 2. NLP Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing NLP Text Processing (TF-IDF + LSA) & Engineering 32 Breakdown Root-Cause Features...")
    featured_df = clean_and_engineer_features(raw_df)

    exception_cnt = int(featured_df['is_breakdown_rca_exception'].sum())
    exception_rate = (exception_cnt / len(featured_df)) * 100.0
    contamination = max(0.05, min(0.45, exception_cnt / max(1, len(featured_df))))
    print(f"[SUMMARY] Total Extracted SAP Plant Maintenance Orders: {len(featured_df):,}")
    print(f"[SUMMARY] Benchmark Critical Breakdown & Unverified Root-Cause Exceptions: {exception_cnt:,} ({exception_rate:.2f}%)")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 32-Feature NLP Breakdown Root-Cause Matrix & Applying RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train & Compare 5 Unsupervised NLP Clustering Models
    print("\n>>> STEP 4: Training 5 Unsupervised NLP Clustering ML Models (Input-Only Training, No Output Labels)...")
    suite = BreakdownRCAClusteringSuite(n_clusters=n_clusters, contamination=contamination, random_state=42)
    suite.train_all(X_train_scaled)

    print("\n>>> STEP 5: Evaluating Accuracy & Ranking the 5 Unsupervised NLP Clustering ML Models on Test Split...")
    test_preds = suite.predict_all(X_test_scaled)
    comparison_df = evaluate_models(test_preds, y_test)

    print("\n" + "=" * 102)
    print("  5 UNSUPERVISED NLP CLUSTERING ML MODELS RANKING & ACCURACY TABLE (TEST SET EVALUATION)")
    print("=" * 102)
    print(comparison_df.to_string(index=False))
    print("=" * 102)

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

    # 6. Fit Selected #1 Champion Model (K-Means Clustering) & Score All Maintenance Orders
    print(f"\n>>> STEP 6: Extracting Root Causes & Scoring All {len(featured_df):,} SAP PM Orders with #1 Champion Model ({champion_model_name})...")
    analyzer = SingleModelBreakdownRCAAnalyzer(
        n_clusters=n_clusters,
        contamination=contamination,
        random_state=42,
        pretrained_model=suite.models['kmeans']
    )
    analyzer.fit(X_train_scaled)
    scored_df = score_with_single_model(featured_df, analyzer, scaler, feature_cols)

    print("\n[NLP EXTRACTED ROOT CAUSE ENGINEERING DOMAIN DISTRIBUTION]:")
    for dom_name, cnt in scored_df['extracted_root_cause_domain'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        print(f"  - {dom_name:<54}: {cnt:>6,} orders ({pct:.2f}%)")

    print("\n[BREAKDOWN ROOT CAUSE CLUSTERING SEGMENT DISTRIBUTION]:")
    for seg_name, cnt in scored_df['rca_segment_name'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        print(f"  - {seg_name:<54}: {cnt:>6,} orders ({pct:.2f}%)")

    crit_cnt = int((scored_df['risk_category'] == 'Critical Risk').sum())
    high_cnt = int((scored_df['risk_category'] == 'High Risk').sum())
    med_cnt = int((scored_df['risk_category'] == 'Medium Risk').sum())
    low_cnt = int((scored_df['risk_category'] == 'Low Risk').sum())

    print("\n[EXECUTIVE BREAKDOWN ROOT CAUSE TRIAGE SUMMARY]:")
    print(f"  - Critical Risk: {crit_cnt:,} orders (PM01 emergency breakdowns, unclosed failures, or work-center mismatches)")
    print(f"  - High Risk:     {high_cnt:,} orders (severe component wear or unverified technician confirmation texts)")
    print(f"  - Medium Risk:   {med_cnt:,} orders (moderate-clarity corrective maintenance & chronic equipment wear)")
    print(f"  - Low Risk:      {low_cnt:,} orders (well-documented preventive & calibrated PM orders with verified RCA)")

    # Save top breakdown root-cause exception watchlist CSV
    scored_df[scored_df['risk_category'].isin(['Critical Risk', 'High Risk', 'Medium Risk'])].head(1500).to_csv(
        "reports/breakdown_rca_watchlist.csv", index=False
    )

    # 7. Compute 32-Feature Importance & Explainability
    print("\n>>> STEP 7: Computing 32-Feature Importance for Breakdown Root Cause Analysis...")
    feat_imp = compute_feature_importance(analyzer.model, feature_cols, X_train_scaled, y_true=y_train)
    print("\n=== TOP 15 FEATURE IMPORTANCE SCORES ===")
    print(feat_imp.head(15).to_string(index=False))
    feat_imp.to_csv("reports/feature_importance.csv", index=False)

    # 8. Export Enriched 3-Sheet Excel Reports
    print("\n>>> STEP 8: Generating Executive 3-Sheet Enriched Excel Report...")
    desc_text = (
        "Extract root causes from maintenance text using Unsupervised NLP (TF-IDF + LSA Semantic "
        "Embeddings + Lexical Failure-Mode Taxonomy) & Clustering across SAP Plant Maintenance "
        "Order Master (AUFK), Order Header (AFKO), Order Confirmations (AFRU), Spare Parts Goods "
        "Movements (AUFM, MAKT), Component Reservations (RESB), and Work Centers (CRHD)."
    )
    excel_path = export_enriched_excel_report(
        df=scored_df,
        output_path=output_excel,
        insight_id="ML_26",
        use_case="Breakdown Root Cause Analysis",
        description=desc_text,
        feature_importance_df=feat_imp
    )

    # Also save standard named files in root and reports/
    import shutil
    for extra_path in [
        "ML_26_Breakdown_Root_Cause_Analysis_Enriched_Output.xlsx",
        os.path.join("reports", "ML_26_Breakdown_Root_Cause_Analysis_Enriched_Output.xlsx"),
        os.path.join("reports", os.path.basename(output_excel))
    ]:
        if os.path.abspath(extra_path) != os.path.abspath(excel_path):
            os.makedirs(os.path.dirname(os.path.abspath(extra_path)), exist_ok=True)
            shutil.copy2(excel_path, extra_path)
            print(f"[SUCCESS] Copied Executive Enriched Excel Report to: {extra_path}")

    # 9. Persist Model Artifacts
    print("\n>>> STEP 9: Persisting Model Pipeline & Benchmark Artifacts...")
    save_single_model_pipeline(
        analyzer, scaler, feature_cols, suite=suite, comparison_df=comparison_df, output_dir="models"
    )

    print("\n" + "=" * 102)
    print("  BREAKDOWN ROOT CAUSE ANALYSIS (NLP) PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  CHAMPION MODEL: {champion_model_name} (Accuracy: {champion_acc:.2f}%, Precision: {champion_prec:.2f}%)")
    print(f"  EXCEL REPORT GENERATED: {os.path.abspath(excel_path)}")
    print("=" * 102)
    return excel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAP Breakdown Root Cause Analysis (NLP) with 5 Unsupervised ML Models & Champion Excel Output")
    parser.add_argument("--limit", type=int, default=None, help="Number of SAP PM order records to extract (default: None for ALL data without limit)")
    parser.add_argument("--clusters", type=int, default=4, help="Number of breakdown root-cause clusters (default: 4)")
    parser.add_argument("--output", type=str, default="ML_26_Breakdown_Root_Cause_Analysis_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, n_clusters=args.clusters, output_excel=args.output)
