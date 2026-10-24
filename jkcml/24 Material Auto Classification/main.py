"""
Main Execution Script - Use Case 24: Material Auto Classification (NLP)
Runs the end-to-end Unsupervised NLP & Clustering Machine Learning pipeline:
1. Connects to PostgreSQL server (192.168.1.44:5432 / JKC_Input / HANA_P2 & HANA_P3) and extracts
   ALL SAP Material Master, Material Descriptions, Plant/MRP, Valuation, Purchase Order Short Texts,
   UoM, and Info Records from MARA, MAKT, MARC, MBEW, EKPO, MARM, EINA, T134T, T001W, T001 with ZERO limits.
2. Performs NLP data cleaning, missing value standardization (blanks for missing text, zeros for metrics),
   TF-IDF + TruncatedSVD semantic vectorization, lexical domain affinity scoring, and engineers 32 domain features.
3. Trains 5 Unsupervised NLP Clustering ML models (K-Means Clustering, Gaussian Mixture Model, Mini-Batch K-Means,
   Bisecting K-Means, BIRCH Clustering) strictly on input features X (no output labels used during training).
4. Checks Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Silhouette Score, and Specificity to rank all 5 models
   and selects the #1 Champion Model based on the ranking.
5. Uses the selected #1 Champion Model (K-Means Clustering) to auto-classify and segment all material records into
   4 classification confidence & taxonomy clusters and scores them with calibrated misclassification exception probabilities,
   binary predictions, and risk categories (Critical, High, Medium, Low Risk).
6. Computes 32-feature importance and exports the executive 3-sheet Excel reports and visual evaluation charts.
"""
import os
import argparse
import pandas as pd
from src.db_connector import load_config, extract_material_classification_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import MaterialClassificationClusteringSuite, SingleModelMaterialAutoClassifier
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
    schemas: list = None,
    n_clusters: int = 4,
    output_excel: str = "ML_24_Material_Auto_Classification_Enriched_Output_Clean.xlsx"
):
    print("=" * 98)
    print("  USE CASE 24: MATERIAL AUTO CLASSIFICATION (NLP) - FULL UNLIMITED SERVER DATASET")
    print("  5 UNSUPERVISED ML MODELS -> #1 CHAMPION MODEL -> EXECUTIVE ENRICHED EXCEL REPORT")
    print("=" * 98)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    target_schemas = schemas or config["database"].get("schemas", ["HANA_P2", "HANA_P3"])
    limit_text = f"limit={limit:,}" if limit and limit > 0 else "NO LIMIT (100% complete dataset)"

    print(f"\n>>> STEP 1: Connecting to PostgreSQL Server (192.168.1.44:5432 / {config['database']['dbname']})...")
    print(f"    Extracting joined data across schemas: {target_schemas} with {limit_text}")
    print("    Tables: MARA, MAKT, MARC, MBEW, EKPO, MARM, EINA, T134T, T001W, T001")
    raw_df = extract_material_classification_data(limit=limit, schemas=target_schemas, config=config)

    # 2. NLP Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing NLP Text Processing (TF-IDF + LSA) & Engineering 32 Material Classification Features...")
    print("    Standardizing missing values: proper blank strings ('') for text and zeros for numerical metrics...")
    featured_df = clean_and_engineer_features(raw_df)

    exception_cnt = int(featured_df['is_classification_exception'].sum())
    exception_rate = (exception_cnt / len(featured_df)) * 100.0
    contamination = max(0.05, min(0.45, exception_cnt / max(1, len(featured_df))))
    print(f"[SUMMARY] Total Extracted SAP Material Records: {len(featured_df):,}")
    print(f"[SUMMARY] Benchmark Misclassified / Ambiguous Material Exceptions: {exception_cnt:,} ({exception_rate:.2f}%)")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 32-Feature NLP Material Classification Matrix & Applying RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train & Compare 5 Unsupervised NLP Clustering Models
    print("\n>>> STEP 4: Training 5 Unsupervised NLP Clustering ML Models (Input-Only Training, No Output Labels)...")
    suite = MaterialClassificationClusteringSuite(n_clusters=n_clusters, contamination=contamination, random_state=42)
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

    # 6. Fit Selected #1 Champion Model (K-Means Clustering) & Auto-Classify All Material Records
    print(f"\n>>> STEP 6: Auto-Classifying & Scoring All {len(featured_df):,} SAP Materials with #1 Champion Model ({champion_model_name})...")
    classifier = SingleModelMaterialAutoClassifier(
        n_clusters=n_clusters,
        contamination=contamination,
        random_state=42,
        pretrained_model=suite.models['kmeans']
    )
    classifier.fit(X_train_scaled)
    scored_df = score_with_single_model(featured_df, classifier, scaler, feature_cols)

    print("\n[NLP AUTO-CLASSIFIED ENTERPRISE MATERIAL DOMAIN DISTRIBUTION]:")
    for dom_name, cnt in scored_df['auto_classified_domain'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        print(f"  - {dom_name:<48}: {cnt:>6,} materials ({pct:.2f}%)")

    print("\n[MATERIAL CLASSIFICATION CLUSTERING SEGMENT DISTRIBUTION]:")
    for seg_name, cnt in scored_df['material_segment_name'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        print(f"  - {seg_name:<48}: {cnt:>6,} materials ({pct:.2f}%)")

    crit_cnt = int((scored_df['risk_category'] == 'Critical Risk').sum())
    high_cnt = int((scored_df['risk_category'] == 'High Risk').sum())
    med_cnt = int((scored_df['risk_category'] == 'Medium Risk').sum())
    low_cnt = int((scored_df['risk_category'] == 'Low Risk').sum())

    print("\n[EXECUTIVE MATERIAL CLASSIFICATION TRIAGE SUMMARY]:")
    print(f"  - Critical Risk: {crit_cnt:,} materials (severe MAKTX vs MTART/MATKL conflicts or cryptic unclassifiable descriptions)")
    print(f"  - High Risk:     {high_cnt:,} materials (ambiguous or unstandardized material group assignments requiring MDM governance)")
    print(f"  - Medium Risk:   {med_cnt:,} materials (moderate-confidence descriptions needing dimensional/spec enrichment)")
    print(f"  - Low Risk:      {low_cnt:,} materials (high-confidence auto-classified materials aligned with SAP taxonomy)")

    # Save top material classification exception watchlist CSV
    watchlist_path = "reports/material_auto_classification_watchlist.csv"
    scored_df[scored_df['risk_category'].isin(['Critical Risk', 'High Risk', 'Medium Risk'])].head(5000).to_csv(
        watchlist_path, index=False
    )
    print(f"[INFO] Saved top material classification watchlist to: {watchlist_path}")

    # 7. Compute 32-Feature Importance & Explainability
    print("\n>>> STEP 7: Computing 32-Feature Importance for Material Auto Classification...")
    # Use representative training sample for feature importance to ensure fast execution
    feat_sample_size = min(35000, len(X_train_scaled))
    rng_feat = pd.Series(range(len(X_train_scaled))).sample(n=feat_sample_size, random_state=42).values
    feat_imp = compute_feature_importance(
        classifier.model, feature_cols, X_train_scaled[rng_feat], y_true=y_train[rng_feat]
    )
    print("\n=== TOP 15 FEATURE IMPORTANCE SCORES ===")
    print(feat_imp.head(15).to_string(index=False))
    feat_imp.to_csv("reports/feature_importance.csv", index=False)

    # 8. Export Enriched 3-Sheet Excel Reports
    print("\n>>> STEP 8: Generating Executive 3-Sheet Enriched Excel Report (Ultra-Fast Streaming with Missing Value Blanks)...")
    desc_text = (
        "Automatically classify materials from descriptions using Unsupervised NLP (TF-IDF + LSA Semantic "
        "Embeddings + Lexical Taxonomy) & Clustering across SAP Material Master (MARA), Material Descriptions "
        "(MAKT), Plant/MRP (MARC), Valuation (MBEW), Purchase Order Short Texts (EKPO), UoM (MARM), and Info Records (EINA) "
        "from HANA_P2 and HANA_P3 with complete server coverage."
    )
    excel_path = export_enriched_excel_report(
        df=scored_df,
        output_path=output_excel,
        insight_id="ML_24",
        use_case="Material Auto Classification",
        description=desc_text,
        feature_importance_df=feat_imp
    )

    # Also save standard named files in root and reports/ via fast copy
    import shutil
    for extra_path in [
        "ML_24_Material_Auto_Classification_Enriched_Output.xlsx",
        os.path.join("reports", "ML_24_Material_Auto_Classification_Enriched_Output.xlsx"),
        os.path.join("reports", os.path.basename(output_excel))
    ]:
        if os.path.abspath(extra_path) != os.path.abspath(excel_path):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(extra_path)), exist_ok=True)
                shutil.copyfile(excel_path, extra_path)
                print(f"[INFO] Copied executive Excel report to: {extra_path}")
            except Exception as e:
                print(f"[WARNING] Could not copy report to {extra_path}: {e}")

    # 9. Persist Model Artifacts
    print("\n>>> STEP 9: Persisting Model Pipeline & Benchmark Artifacts...")
    save_single_model_pipeline(
        classifier, scaler, feature_cols, suite=suite, comparison_df=comparison_df, output_dir="models"
    )

    print("\n" + "=" * 98)
    print("  MATERIAL AUTO CLASSIFICATION (NLP) PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  TOTAL RECORDS PROCESSED: {len(scored_df):,}")
    print(f"  CHAMPION MODEL: {champion_model_name} (Accuracy: {champion_acc:.2f}%, Precision: {champion_prec:.2f}%)")
    print(f"  EXCEL REPORT GENERATED: {os.path.abspath(excel_path)}")
    print("=" * 98)
    return excel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAP Material Auto Classification (NLP) with 5 Unsupervised ML Models & Champion Excel Output")
    parser.add_argument("--limit", type=int, default=None, help="Number of material records to extract (default: None for 100% all data)")
    parser.add_argument("--schemas", nargs="+", default=None, help="Target schemas to query (default: HANA_P2 HANA_P3)")
    parser.add_argument("--clusters", type=int, default=4, help="Number of material classification clusters (default: 4)")
    parser.add_argument("--output", type=str, default="ML_24_Material_Auto_Classification_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, schemas=args.schemas, n_clusters=args.clusters, output_excel=args.output)
