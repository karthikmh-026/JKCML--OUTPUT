"""
Main Execution Script - Use Case 25: Account Auto Coding (NLP)
Runs the end-to-end Unsupervised NLP & Clustering Machine Learning pipeline:
1. Connects to PostgreSQL server (192.168.1.44:5432 / JKC_Input / HANA_P2) and extracts SAP Accounting
   Document Line Items, Transaction Texts, G/L Account Master Data, G/L Account Descriptions,
   Posting Keys, Document Types, PO Short Texts, and Vendor Metadata from BSEG, SKA1, SKAT, SKB1,
   TBSL, T003T, EKPO, LFA1, T001W, and T001.
2. Performs NLP data cleaning, TF-IDF + TruncatedSVD semantic vectorization, lexical G/L domain
   affinity scoring, and engineers 32 domain-specific features to predict G/L accounts from transaction text.
3. Trains 5 Unsupervised NLP Clustering ML models (K-Means Clustering, Gaussian Mixture Model, Mini-Batch K-Means,
   Bisecting K-Means, BIRCH Clustering) strictly on input features X (no output labels used during training).
4. Checks Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Silhouette Score, and Specificity to rank all 5 models
   and selects the #1 Champion Model based on the ranking.
5. Uses the selected #1 Champion Model (K-Means Clustering) to auto-code and segment all extracted financial transaction
   line items into 4 G/L coding confidence & Chart-of-Accounts clusters and scores them with calibrated miscoding exception
   probabilities, binary predictions, and risk categories (Critical, High, Medium, Low Risk).
6. Computes 32-feature importance and exports the executive 3-sheet Excel reports and visual evaluation charts.
"""
import os
import argparse
import pandas as pd
from src.db_connector import load_config, extract_account_auto_coding_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import AccountCodingClusteringSuite, SingleModelAccountAutoCoder
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
    output_excel: str = "ML_25_Account_Auto_Coding_Enriched_Output_Clean.xlsx"
):
    print("=" * 98)
    print("  USE CASE 25: ACCOUNT AUTO CODING (NLP) - 5 UNSUPERVISED ML MODELS -> #1 CHAMPION")
    print("=" * 98)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    print("\n>>> STEP 1: Connecting to PostgreSQL Server (192.168.1.44:5432) & Extracting SAP FI Transactions...")
    print("    Extracting joined data from: BSEG, SKA1, SKAT, SKB1, TBSL, T003T, EKPO, LFA1, T001W, T001")
    raw_df = extract_account_auto_coding_data(limit=limit, config=config)

    # 2. NLP Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing NLP Text Processing (TF-IDF + LSA) & Engineering 32 GL Account Auto-Coding Features...")
    featured_df = clean_and_engineer_features(raw_df)

    exception_cnt = int(featured_df['is_coding_exception'].sum())
    exception_rate = (exception_cnt / len(featured_df)) * 100.0
    contamination = max(0.05, min(0.45, exception_cnt / max(1, len(featured_df))))
    print(f"[SUMMARY] Total Extracted SAP Accounting Line Items: {len(featured_df):,}")
    print(f"[SUMMARY] Benchmark Miscoded / Ambiguous G/L Coding Exceptions: {exception_cnt:,} ({exception_rate:.2f}%)")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 32-Feature NLP G/L Account Auto-Coding Matrix & Applying RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train & Compare 5 Unsupervised NLP Clustering Models
    print("\n>>> STEP 4: Training 5 Unsupervised NLP Clustering ML Models (Input-Only Training, No Output Labels)...")
    suite = AccountCodingClusteringSuite(n_clusters=n_clusters, contamination=contamination, random_state=42)
    suite.train_all(X_train_scaled)

    print("\n>>> STEP 5: Evaluating Accuracy & Ranking the 5 Unsupervised NLP Clustering ML Models on Test Split...")
    test_preds = suite.predict_all(X_test_scaled)
    comparison_df = evaluate_models(test_preds, y_test)

    print("\n" + "=" * 98)
    print("  5 UNSUPERVISED NLP CLUSTERING ML MODELS RANKING & ACCURACY TABLE (TEST SET EVALUATION)")
    print("=" * 98)
    print(comparison_df.to_string(index=False))
    print("=" * 98)

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

    # 6. Fit Selected #1 Champion Model (K-Means Clustering) & Auto-Code All Financial Transaction Records
    print(f"\n>>> STEP 6: Predicting GL Accounts & Scoring All {len(featured_df):,} SAP Transactions with #1 Champion Model ({champion_model_name})...")
    name_to_key = {
        'K-Means Clustering': 'kmeans',
        'Gaussian Mixture (GMM)': 'gmm',
        'Mini-Batch K-Means': 'minibatch_kmeans',
        'Bisecting K-Means': 'bisecting_kmeans',
        'BIRCH Clustering': 'birch'
    }
    champ_key = name_to_key.get(champion_model_name, 'kmeans')
    coder = SingleModelAccountAutoCoder(
        n_clusters=n_clusters,
        contamination=contamination,
        random_state=42,
        pretrained_model=suite.models.get(champ_key, suite.models.get('kmeans'))
    )
    coder.fit(X_train_scaled)
    scored_df = score_with_single_model(featured_df, coder, scaler, feature_cols)

    print("\n[NLP PREDICTED ENTERPRISE ACCOUNTING DOMAIN DISTRIBUTION]:")
    for dom_name, cnt in scored_df['predicted_gl_domain'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        print(f"  - {dom_name:<52}: {cnt:>6,} postings ({pct:.2f}%)")

    print("\n[ACCOUNT AUTO-CODING CLUSTERING SEGMENT DISTRIBUTION]:")
    for seg_name, cnt in scored_df['coding_segment_name'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        print(f"  - {seg_name:<52}: {cnt:>6,} postings ({pct:.2f}%)")

    crit_cnt = int((scored_df['risk_category'] == 'Critical Risk').sum())
    high_cnt = int((scored_df['risk_category'] == 'High Risk').sum())
    med_cnt = int((scored_df['risk_category'] == 'Medium Risk').sum())
    low_cnt = int((scored_df['risk_category'] == 'Low Risk').sum())

    print("\n[EXECUTIVE GL ACCOUNT AUTO-CODING TRIAGE SUMMARY]:")
    print(f"  - Critical Risk: {crit_cnt:,} postings (severe BSEG-SGTXT vs HKONT conflicts or cryptic 2-char narrations)")
    print(f"  - High Risk:     {high_cnt:,} postings (specific operational expenses parked in generic provision/clearing GLs)")
    print(f"  - Medium Risk:   {med_cnt:,} postings (moderate-confidence narrations needing cost center or text enrichment)")
    print(f"  - Low Risk:      {low_cnt:,} postings (high-confidence auto-coded GL postings aligned with SAP Chart of Accounts)")

    # Save top account auto-coding exception watchlist CSV
    scored_df[scored_df['risk_category'].isin(['Critical Risk', 'High Risk', 'Medium Risk'])].head(1500).to_csv(
        "reports/account_auto_coding_watchlist.csv", index=False
    )

    # 7. Compute 32-Feature Importance & Explainability
    print("\n>>> STEP 7: Computing 32-Feature Importance for Account Auto Coding...")
    feat_imp = compute_feature_importance(coder.model, feature_cols, X_train_scaled, y_true=y_train)
    print("\n=== TOP 15 FEATURE IMPORTANCE SCORES ===")
    print(feat_imp.head(15).to_string(index=False))
    feat_imp.to_csv("reports/feature_importance.csv", index=False)

    # 8. Export Enriched 3-Sheet Excel Reports
    print("\n>>> STEP 8: Generating Executive 3-Sheet Enriched Excel Report...")
    desc_text = (
        "Predict G/L accounts from transaction text using Unsupervised NLP (TF-IDF + LSA Semantic "
        "Embeddings + Lexical Chart-of-Accounts Taxonomy) & Clustering across SAP Accounting Document "
        "Segments (BSEG), G/L Account Master (SKA1, SKAT, SKB1), Posting Keys (TBSL), Document Types (T003T), "
        "and Purchase Order Short Texts (EKPO)."
    )
    excel_path = export_enriched_excel_report(
        df=scored_df,
        output_path=output_excel,
        insight_id="ML_25",
        use_case="Account Auto Coding",
        description=desc_text,
        feature_importance_df=feat_imp
    )

    # Also save standard named copies in root and reports/
    import shutil
    for extra_path in [
        "ML_25_Account_Auto_Coding_Enriched_Output.xlsx",
        os.path.join("reports", "ML_25_Account_Auto_Coding_Enriched_Output.xlsx"),
        os.path.join("reports", os.path.basename(output_excel))
    ]:
        if os.path.abspath(extra_path) != os.path.abspath(excel_path):
            os.makedirs(os.path.dirname(os.path.abspath(extra_path)), exist_ok=True)
            shutil.copyfile(excel_path, extra_path)

    # 9. Persist Model Artifacts
    print("\n>>> STEP 9: Persisting Model Pipeline & Benchmark Artifacts...")
    save_single_model_pipeline(
        coder, scaler, feature_cols, suite=suite, comparison_df=comparison_df, output_dir="models"
    )

    print("\n" + "=" * 98)
    print("  ACCOUNT AUTO CODING (NLP) PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  CHAMPION MODEL: {champion_model_name} (Accuracy: {champion_acc:.2f}%, Precision: {champion_prec:.2f}%)")
    print(f"  EXCEL REPORT GENERATED: {os.path.abspath(excel_path)}")
    print("=" * 98)
    return excel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAP Account Auto Coding (NLP) with 5 Unsupervised ML Models & Champion Excel Output")
    parser.add_argument("--limit", type=int, default=None, help="Number of transaction records to extract (default: None for full dataset)")
    parser.add_argument("--clusters", type=int, default=4, help="Number of account auto-coding clusters (default: 4)")
    parser.add_argument("--output", type=str, default="ML_25_Account_Auto_Coding_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, n_clusters=args.clusters, output_excel=args.output)
