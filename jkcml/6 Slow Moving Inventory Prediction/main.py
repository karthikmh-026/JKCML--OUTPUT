"""
Main Execution Script - Use Case 6: Slow Moving Inventory Prediction
Runs the end-to-end Unsupervised Machine Learning pipeline on ALL data without limits:
1. Connects to PostgreSQL database (192.168.1.44:5432 / JKC_Input)
   and extracts ALL material inventory records from BOTH HANA_P2 and HANA_P3 schemas.
2. Cleans and engineers 32 domain features for inventory velocity, turnover, and dormancy.
   Handles missing values cleanly (blank "" for missing text, proper numeric imputation).
3. Trains 5 Unsupervised Machine Learning Models strictly on input features (no target labels used).
4. Evaluates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Silhouette Score to rank all 5 models.
5. Selects the #1 Ranked Champion Model based on the ranking.
6. Uses the selected Champion Model to segment all materials into 4 Inventory Velocity Tiers and calculate
   calibrated Probability of Obsolescence P(Slow Moving), binary predictions, and risk categories.
7. Persists champion model artifacts and exports executive 3-sheet Excel reports and diagnostic charts.
"""
import os
import shutil
import argparse
import pandas as pd
from src.db_connector import load_config, extract_slow_moving_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import InventoryClusteringSuite, SingleModelInventorySegmenter
from src.evaluate import (
    evaluate_models,
    plot_model_comparison,
    plot_roc_pr_curves,
    plot_inventory_distribution,
    compute_feature_importance,
    plot_feature_importance
)
from src.inference import (
    score_with_single_model,
    export_enriched_excel_report,
    save_single_model_pipeline
)


def run_pipeline(
    limit: int = None,
    n_clusters: int = 4,
    output_excel: str = "ML_6_Slow_Moving_Inventory_Prediction_Enriched_Output_Clean.xlsx"
):
    print("=" * 96)
    print("  USE CASE 6: SLOW MOVING INVENTORY PREDICTION & OBSOLESCENCE IDENTIFICATION")
    print("  FULL SCALE (HANA_P2 + HANA_P3) -> 5 UNSUPERVISED ML MODELS -> RANKING -> #1 CHAMPION SCORING")
    print("=" * 96)

    # 1. Config & Data Extraction directly from PostgreSQL Server (No Limit)
    config = load_config()
    print("\n>>> STEP 1: Connecting to PostgreSQL (192.168.1.44:5432 / JKC_Input) & Extracting ALL SAP Inventory Data...")
    print("    Extracting joined data across BOTH HANA_P2 and HANA_P3 (MBEW, MARA, MAKT, MARC, T001W, MSEG)...")
    raw_df = extract_slow_moving_data(limit=limit, config=config, use_cache=True)

    # 2. Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing Data Cleaning & Engineering 32 Inventory Velocity & Inactivity Features...")
    print("    Handling missing values: text fields left blank (''), numerics properly imputed...")
    featured_df = clean_and_engineer_features(raw_df)

    exception_cnt = int(featured_df['is_slow_moving_exception'].sum())
    exception_rate = (exception_cnt / max(1, len(featured_df))) * 100.0
    contamination = max(0.10, min(0.60, exception_cnt / max(1, len(featured_df))))
    print(f"[SUMMARY] Total Extracted SAP Positive-Stock Records: {len(featured_df):,}")
    print(f"[SUMMARY] Breakdown by Schema: {featured_df['source_schema'].value_counts().to_dict()}")
    print(f"[SUMMARY] Benchmark Slow-Moving / Obsolete Records: {exception_cnt:,} ({exception_rate:.2f}%)")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 32-Feature Inventory Matrix & Applying RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train 5 Unsupervised ML Models
    print("\n>>> STEP 4: Training 5 Unsupervised ML Models (Input-Only Training, No Output Labels Used)...")
    suite = InventoryClusteringSuite(n_clusters=n_clusters, contamination=contamination, random_state=42)
    suite.train_all(X_train_scaled)

    # 5. Evaluate & Rank 5 Models
    print("\n>>> STEP 5: Evaluating Accuracy & Ranking All 5 Unsupervised Models on Held-out Test Split...")
    test_preds = suite.predict_all(X_test_scaled)
    comparison_df = evaluate_models(test_preds, y_test)

    print("\n" + "=" * 96)
    print("  5 UNSUPERVISED ML MODELS RANKING & ACCURACY TABLE (TEST SET EVALUATION)")
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
    champion_rec = comparison_df.iloc[0]['Recall'] * 100.0
    champion_f1 = comparison_df.iloc[0]['F1-Score']
    champion_auc = comparison_df.iloc[0]['ROC-AUC']
    champion_sil = comparison_df.iloc[0]['Silhouette']
    print(f"\n[CHAMPION SELECTED] Rank #1 Model: {champion_model_name}")
    print(f"  -> Accuracy: {champion_acc:.2f}% | Precision: {champion_prec:.2f}% | Recall: {champion_rec:.2f}% | F1-Score: {champion_f1:.4f} | ROC-AUC: {champion_auc:.4f} | Silhouette: {champion_sil:.4f}")

    # 6. Fit Selected #1 Champion Model & Segment All Inventory
    print(f"\n>>> STEP 6: Deploying Selected #1 Champion Model ({champion_model_name}) for Production Scoring...")
    champion_segmenter = SingleModelInventorySegmenter(
        model_name=champion_model_name,
        n_clusters=n_clusters,
        contamination=contamination,
        random_state=42
    )
    champion_segmenter.fit(X_train_scaled)

    # 7. Score All Materials & Generate Recommendations
    print("\n>>> STEP 7: Scoring All Material Records & Estimating Probability of Obsolescence P(Slow Moving)...")
    scored_df = score_with_single_model(champion_segmenter, featured_df, scaler, feature_cols)

    # 8. Feature Importance
    print("\n>>> STEP 8: Computing 32-Feature Importance and Inventory Explainability...")
    fi_sample_size = min(50000, len(X_train_scaled))
    fi_df = compute_feature_importance(X_train_scaled[:fi_sample_size], y_train[:fi_sample_size], feature_cols)
    fi_df.to_csv("reports/feature_importance.csv", index=False)
    plot_feature_importance(fi_df, output_dir="reports/figures", top_n=15)
    print("Top 10 Most Influential Obsolescence Drivers:")
    print(fi_df.head(10)[['Rank', 'Feature', 'Importance', 'Relative_Weight_Pct']].to_string(index=False))

    # 9. Generate Charts & Excel Reports
    print("\n>>> STEP 9: Generating Distribution Visualizations & Multi-Sheet Executive Excel Reports...")
    plot_inventory_distribution(scored_df, output_dir="reports/figures")
    export_enriched_excel_report(scored_df, output_path=output_excel, fi_df=fi_df, comp_df=comparison_df)

    alt_excel = "ML_6_Slow_Moving_Inventory_Prediction_Enriched_Output.xlsx"
    shutil.copyfile(output_excel, alt_excel)
    print(f"[REPORT] Copied enriched Excel report to {alt_excel}")

    # 10. Persist Model Artifacts
    print("\n>>> STEP 10: Persisting Champion Model Pipeline & Scaler Artifacts to models/...")
    save_single_model_pipeline(champion_segmenter, scaler, feature_cols, output_dir="models")

    print("\n" + "=" * 96)
    print("  PIPELINE EXECUTION COMPLETE - ALL DELIVERABLES GENERATED SUCCESSFULLY!")
    print(f"  * Total Materials Scored: {len(scored_df):,}")
    print(f"  * #1 Champion Model: {champion_model_name}")
    print(f"  * Model Comparison Metrics: reports/model_comparison_metrics.csv")
    print(f"  * Feature Importance: reports/feature_importance.csv")
    print(f"  * Visual Figures: reports/figures/ (model_comparison.png, roc_pr_curves.png, cluster_distribution.png)")
    print(f"  * Model Artifacts: models/ (champion_slow_moving_model.joblib, slow_moving_model_bundle.joblib)")
    print(f"  * Executive Excel Reports: {output_excel}")
    print("=" * 96)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Case 6: Slow Moving Inventory Prediction")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of inventory records to extract (default: None = all data)")
    parser.add_argument("--clusters", type=int, default=4, help="Number of inventory velocity tier clusters")
    parser.add_argument("--output", type=str, default="ML_6_Slow_Moving_Inventory_Prediction_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, n_clusters=args.clusters, output_excel=args.output)
