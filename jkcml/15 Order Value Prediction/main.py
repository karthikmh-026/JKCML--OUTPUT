"""
Main Execution Script - Use Case 15: Order Value Prediction
Category: Regression
Objective: Predict next order value of customers.

Runs the end-to-end Supervised Machine Learning Regression pipeline across ALL data from PostgreSQL:
1. Connects to PostgreSQL server (192.168.1.44:5432 / JKC_Input / HANA_P2 & HANA_P3) and extracts live SAP Sales Orders,
   Aggregated Line Items, and Customer Master records from VBAK, VBAP, KNA1, KNVV without artificial row limits.
2. Performs data cleaning, constructs customer order history, defines target (next_order_value),
   and engineers 32 domain-specific features using vectorized pandas/numpy operations.
3. Trains 5 Supervised Regression Models (Random Forest, Gradient Boosting, HistGradientBoosting,
   Decision Tree, Ridge) on 80/20 train/test split.
4. Evaluates R2 Score, MAE, RMSE, MAPE, Accuracy (±20%), and Pearson r to rank all 5 models
   and selects the #1 Champion Regression Model.
5. Uses the selected #1 Champion Model to score all sales orders with:
   - Predicted next order values (INR)
   - Actual next order values (where observed)
   - Prediction variance amounts and percentages
   - Accuracy (±20% Band) flags
   - Expected customer growth trends (Expansion vs Contraction Risk)
   - Strategic predicted value tiers (Enterprise, Commercial, Growth, Retail).
6. Computes 32-feature importance, generates regression diagnostic charts, saves the champion model,
   and exports the executive multi-sheet Excel reports (with strict blank-cell missing value handling)
   and the complete multi-million-row scored dataset in CSV format.
"""
import os
import argparse
import pandas as pd
from src.db_connector import load_config, extract_order_value_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import OrderValueRegressionSuite, SingleModelOrderValueRegressor
from src.evaluate import (
    evaluate_models,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residuals_distribution,
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
    output_excel: str = "ML_15_Order_Value_Prediction_Enriched_Output_Clean.xlsx"
):
    print("=" * 96)
    print("  USE CASE 15: ORDER VALUE PREDICTION (CATEGORY: REGRESSION)")
    print("  OBJECTIVE: PREDICT NEXT ORDER VALUE OF CUSTOMERS - 5 ML MODELS -> #1 CHAMPION SELECTION")
    print("  DATABASE: 192.168.1.44:5432 / JKC_Input (SCHEMAS: HANA_P2 & HANA_P3)")
    print("=" * 96)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    print("\n>>> STEP 1: Connecting to PostgreSQL Server (192.168.1.44:5432 / HANA_P2 & HANA_P3) & Extracting SAP Sales Orders...")
    print("    Extracting joined data from: VBAK, VBAP, KNA1, KNVV")
    raw_df = extract_order_value_data(limit=limit, config=config)

    # 2. Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing Data Cleaning, Constructing Target & Engineering 32 Domain Features...")
    featured_df = clean_and_engineer_features(raw_df)

    observed_next_cnt = int(featured_df['next_order_value'].notna().sum())
    observed_pct = (observed_next_cnt / len(featured_df)) * 100.0
    print(f"[SUMMARY] Total Extracted SAP Sales Orders: {len(featured_df):,}")
    print(f"[SUMMARY] Unique Customers Identified: {featured_df['customer_id'].nunique():,}")
    print(f"[SUMMARY] Trainable Sequential Order Pairs: {observed_next_cnt:,} ({observed_pct:.1f}%)")
    print(f"[SUMMARY] Upcoming Next Orders to Forecast: {len(featured_df) - observed_next_cnt:,} ({100.0 - observed_pct:.1f}%)")

    # 3. Data Preparation & Robust Scaling
    print("\n>>> STEP 3: Preparing 32-Feature Matrix & 80/20 Train/Test Split (RobustScaler)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train 5 Supervised Machine Learning Regression Models
    print("\n>>> STEP 4: Training 5 Supervised Machine Learning Regression Models...")
    suite = OrderValueRegressionSuite(random_state=42)
    suite.train_all(X_train_scaled, y_train)

    # 5. Evaluate Accuracy & Rank the 5 Models on Held-out Test Split
    print("\n>>> STEP 5: Evaluating R2 Score, MAE, RMSE, MAPE, and Accuracy on Test Split...")
    test_preds = suite.predict_all(X_test_scaled)
    comparison_df = evaluate_models(test_preds, y_test)

    print("\n" + "=" * 96)
    print("  5 SUPERVISED REGRESSION ML MODELS RANKING TABLE (TEST SET EVALUATION)")
    print("=" * 96)
    print(comparison_df.to_string(index=False))
    print("=" * 96)

    os.makedirs("reports/figures", exist_ok=True)
    comparison_df.to_csv("reports/model_comparison_metrics.csv", index=False)
    plot_model_comparison(comparison_df, output_dir="reports/figures")

    champion_model_name = comparison_df.iloc[0]['Model']
    champion_key = comparison_df.iloc[0]['model_key']
    champion_r2 = comparison_df.iloc[0]['R2 Score']
    champion_mae = comparison_df.iloc[0]['MAE (INR)']
    champion_acc = comparison_df.iloc[0]['Accuracy (±20%)']
    champion_corr = comparison_df.iloc[0]['Pearson r']
    print(f"\n[CHAMPION SELECTED] Rank #1 Model: {champion_model_name}")
    print(f"  -> R2 Score: {champion_r2:.4f} | MAE: INR {champion_mae:,.2f} | Accuracy (+-20%): {champion_acc}% | Pearson r: {champion_corr:.4f}")

    # Plot Actual vs Predicted & Residuals for Champion Model
    champ_test_pred = test_preds[champion_key]
    plot_actual_vs_predicted(y_test, champ_test_pred, champion_model_name, output_dir="reports/figures")
    plot_residuals_distribution(y_test, champ_test_pred, champion_model_name, output_dir="reports/figures")

    # 6. Fit Selected #1 Champion Model & Score All Orders
    print(f"\n>>> STEP 6: Scoring All {len(featured_df):,} SAP Sales Orders with #1 Champion Model ({champion_model_name})...")
    regressor = SingleModelOrderValueRegressor(
        random_state=42,
        pretrained_model=suite.models[champion_key]
    )
    scored_df = score_with_single_model(featured_df, regressor, scaler, feature_cols)

    print("\n[EXECUTIVE VALUE TIER BREAKDOWN (PREDICTED NEXT ORDER VALUE)]:")
    for tier, cnt in scored_df['predicted_value_tier'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        val = scored_df[scored_df['predicted_value_tier'] == tier]['predicted_next_order_value'].sum()
        print(f"  - {tier:<45}: {cnt:>6,} orders ({pct:.2f}%) | Total INR {val:,.2f}")

    print("\n[CUSTOMER EXPECTED GROWTH TREND BREAKDOWN]:")
    for trend, cnt in scored_df['expected_growth_trend'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        print(f"  - {trend:<45}: {cnt:>6,} orders ({pct:.2f}%)")

    # Save top customer next order forecast watchlist CSV
    os.makedirs("reports", exist_ok=True)
    watchlist_cols = [
        'sales_order_id', 'order_date', 'customer_id', 'customer_name', 'customer_city', 'customer_region',
        'order_net_value', 'predicted_next_order_value', 'actual_next_order_value',
        'prediction_variance_amount', 'prediction_variance_pct', 'is_accurate_within_20pct',
        'expected_growth_trend', 'predicted_value_tier'
    ]
    avail_cols = [c for c in watchlist_cols if c in scored_df.columns]
    scored_df[avail_cols].head(5000).to_csv("reports/order_value_prediction_watchlist.csv", index=False)
    print(f"[INFO] Exported forecast watchlist to reports/order_value_prediction_watchlist.csv")

    # 7. Compute 32-Feature Importance & Explainability
    print("\n>>> STEP 7: Computing 32-Feature Importance for Order Value Prediction...")
    feat_imp = compute_feature_importance(regressor.model, feature_cols)
    print("\n=== TOP 15 FEATURE IMPORTANCE SCORES ===")
    print(feat_imp.head(15).to_string(index=False))
    feat_imp.to_csv("reports/feature_importance.csv", index=False)
    plot_feature_importance(feat_imp, output_dir="reports/figures")

    # 8. Export Enriched Multi-Sheet Excel Reports with Strict Missing Value Sanitization
    print("\n>>> STEP 8: Generating Executive Multi-Sheet Enriched Excel Report with Feature_Importance & Missing Value Sanitization...")
    desc_text = (
        "Predict next order value of customers using Supervised Machine Learning Regression "
        "across SAP Sales Order Header (VBAK), Line Items (VBAP), and Customer Master (KNA1, KNVV)."
    )

    excel_path = export_enriched_excel_report(
        df=scored_df,
        output_path=output_excel,
        insight_id="ML_15",
        use_case="Order Value Prediction",
        description=desc_text,
        evaluation_df=comparison_df,
        feature_importance_df=feat_imp
    )

    # Also save standard copy
    std_copy = "ML_15_Order_Value_Prediction_Enriched_Output.xlsx"
    if std_copy != output_excel:
        export_enriched_excel_report(
            df=scored_df,
            output_path=std_copy,
            insight_id="ML_15",
            use_case="Order Value Prediction",
            description=desc_text,
            evaluation_df=comparison_df,
            feature_importance_df=feat_imp
        )

    # Save champion model artifact
    save_single_model_pipeline(regressor, "models/champion_order_value_regressor.joblib")

    print("\n" + "=" * 96)
    print("  USE CASE 15 REGRESSION PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Enriched Excel Report: {output_excel}")
    print(f"  Full Scored CSV:       ML_15_Order_Value_Prediction_Enriched_Output_Full.csv")
    print(f"  Model Comparison CSV:  reports/model_comparison_metrics.csv")
    print(f"  Feature Importance:    reports/feature_importance.csv")
    print(f"  Champion Model:        models/champion_order_value_regressor.joblib")
    print("=" * 96)

    return scored_df, comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Case 15: Order Value Prediction Regression Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Row extraction limit from PostgreSQL (None for all data)")
    parser.add_argument("--output", type=str, default="ML_15_Order_Value_Prediction_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, output_excel=args.output)
