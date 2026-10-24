"""
Main Execution Script - Use Case 13: Cost Center Spend Forecasting (Time Series)
Runs the end-to-end Time-Series Machine Learning forecasting pipeline:
1. Connects to PostgreSQL server (192.168.1.44:5432 / JKC_Input / HANA_P3) and extracts SAP
   Controlling & Financial Accounting Cost Center Spend Data across 2025 and 2026 from RSEG,
   RBKP, EKKN, CSKS, SKAT, SKA1, AUFK, T001W, T001, and LFA1.
2. Performs Time-Series multi-scale lag & rolling window feature engineering, departmental spend
   categorization, and calendar harmonic Fourier features.
3. Partitions data chronologically: 2025 Historical Data (Training Set) -> 2026 Data (Forecast Horizon).
4. Trains 5 Time-Series ML Models:
   - Random Forest Time-Series Forecaster
   - Gradient Boosting Time-Series Forecaster
   - Ridge Time-Series Forecaster
   - Holt-Winters Exponential Smoothing Forecaster
   - Bayesian AutoRegressive Forecaster
5. Evaluates 2026 forecasting accuracy (MAE, RMSE, MAPE, Accuracy %, R² Score, Forecast Bias %),
   ranks all 5 models, and selects the #1 Champion Model.
6. Scores all departmental postings with the #1 Champion Model, computes spend forecast variances,
   individual forecast accuracy %, continuous overrun probabilities, and executive risk tiers.
7. Computes 32-feature importance and exports executive 3-sheet Excel reports and visual evaluation charts.
"""
import os
import shutil
import argparse
import pandas as pd
from src.db_connector import load_config, extract_cost_center_spend_data
from src.preprocessing import clean_and_engineer_features, prepare_time_series_train_test
from src.models import CostCenterSpendForecastingSuite, SingleModelCostCenterSpendForecaster
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
    output_excel: str = "ML_13_Cost_Center_Spend_Forecasting_Enriched_Output_Clean.xlsx"
):
    print("=" * 105)
    print("  USE CASE 13: COST CENTER SPEND FORECASTING (TIME SERIES: 2025 HISTORY -> 2026 FORECAST)")
    print("=" * 105)

    # 1. Config & Data Extraction from PostgreSQL Server
    config = load_config()
    schema = config["database"].get("schema", "HANA_P3")
    ts_params = config.get("time_series_parameters", {})
    train_year = int(ts_params.get("train_year", 2025))
    forecast_year = int(ts_params.get("forecast_year", 2026))

    print(f"\n>>> STEP 1: Connecting to PostgreSQL Server (192.168.1.44:5432 / {schema}) & Extracting {train_year}-{forecast_year} Spend Data...")
    print("    Extracting joined data from: RSEG, RBKP, EKKN, CSKS, SKAT, SKA1, AUFK, T001W, T001, LFA1")
    raw_df = extract_cost_center_spend_data(config=config)

    # 2. Time-Series Data Cleaning & Feature Engineering
    print("\n>>> STEP 2: Performing Time-Series Lag/Rolling Feature Engineering & Departmental Spend Categorization...")
    featured_df = clean_and_engineer_features(raw_df)

    n_train = (featured_df['posting_year'] == train_year).sum()
    n_forecast = (featured_df['posting_year'] == forecast_year).sum()
    print(f"[SUMMARY] Total Extracted Records:             {len(featured_df):,}")
    print(f"[SUMMARY] {train_year} Historical Training Records:     {n_train:,} ({n_train/len(featured_df)*100:.1f}%)")
    print(f"[SUMMARY] {forecast_year} Forecast Horizon Records:        {n_forecast:,} ({n_forecast/len(featured_df)*100:.1f}%)")
    print(f"[SUMMARY] Distinct Active Cost Centers:         {featured_df['cost_center'].nunique():,}")

    # 3. Chronological Time-Series Train/Test Split (2025 Train -> 2026 Test)
    print(f"\n>>> STEP 3: Partitioning Chronological Train Set ({train_year}) and Forecast Test Horizon ({forecast_year})...")
    data_dict = prepare_time_series_train_test(featured_df, train_year=train_year, test_year=forecast_year)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    print(f"  -> X_train shape: {X_train_scaled.shape} ({train_year} Training Set)")
    print(f"  -> X_test shape:  {X_test_scaled.shape} ({forecast_year} Forecast Horizon)")

    # 4. Train 5 Time-Series ML Models
    print(f"\n>>> STEP 4: Training 5 Time-Series Machine Learning Models on {train_year} Historical Data...")
    suite = CostCenterSpendForecastingSuite(random_state=42)
    suite.train_all(X_train_scaled, y_train)

    # 5. Predict on 2026 Test Horizon & Evaluate Accuracy Ranking
    print(f"\n>>> STEP 5: Forecasting {forecast_year} Expenditure & Evaluating Accuracy Metrics across all 5 Models...")
    test_preds = suite.predict_all(X_test_scaled)
    comparison_df = evaluate_models(test_preds, y_test)

    print("\n" + "=" * 105)
    print(f"  5 TIME-SERIES FORECASTING ML MODELS ACCURACY & PERFORMANCE RANKING ({forecast_year} TEST HORIZON)")
    print("=" * 105)
    print(comparison_df.to_string(index=False))
    print("=" * 105)

    os.makedirs("reports/figures", exist_ok=True)
    comparison_df.to_csv("reports/model_comparison_metrics.csv", index=False)
    plot_model_comparison(comparison_df, output_dir="reports/figures")
    plot_roc_pr_curves(test_preds, y_test, output_dir="reports/figures")

    champion_model_name = comparison_df.iloc[0]['Model']
    champion_acc = comparison_df.iloc[0]['Accuracy (%)']
    champion_r2 = comparison_df.iloc[0]['R² Score']
    champion_mae = comparison_df.iloc[0]['MAE (INR)']
    champion_rmse = comparison_df.iloc[0]['RMSE (INR)']
    champion_mape = comparison_df.iloc[0]['MAPE (%)']
    print(f"\n[CHAMPION SELECTED] Rank #1 Model: {champion_model_name}")
    print(f"  -> Forecast Accuracy: {champion_acc:.2f}% | R² Score: {champion_r2:.4f} | MAE: INR {champion_mae:,.2f} | RMSE: INR {champion_rmse:,.2f} | MAPE: {champion_mape:.2f}%")

    # 6. Fit Champion Model & Score All Departmental Spend Postings
    print(f"\n>>> STEP 6: Forecasting Departmental Spending & Scoring All {len(featured_df):,} Records with Champion Model ({champion_model_name})...")
    forecaster = SingleModelCostCenterSpendForecaster(
        champion_name=champion_model_name,
        pretrained_model=suite.models[champion_model_name]
    )
    scored_df = score_with_single_model(featured_df, forecaster, scaler, feature_cols)

    print("\n[DEPARTMENTAL SPEND CATEGORY DISTRIBUTION]:")
    for dom_name, cnt in scored_df['department_spend_category'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        tot_spend = scored_df[scored_df['department_spend_category'] == dom_name]['actual_spend_inr'].sum()
        print(f"  - {dom_name:<46}: {cnt:>6,} postings ({pct:>5.1f}%) | Total Spend: INR {tot_spend:,.2f}")

    crit_cnt = int((scored_df['risk_category'] == 'Critical Risk').sum())
    high_cnt = int((scored_df['risk_category'] == 'High Risk').sum())
    med_cnt = int((scored_df['risk_category'] == 'Medium Risk').sum())
    low_cnt = int((scored_df['risk_category'] == 'Low Risk').sum())

    print("\n[EXECUTIVE COST CENTER SPEND VARIANCE TRIAGE SUMMARY]:")
    print(f"  - Critical Risk: {crit_cnt:>5,} postings (Severe spend overrun: variance >= +50% and spend >= INR 50,000)")
    print(f"  - High Risk:     {high_cnt:>5,} postings (Substantial spend overrun: variance >= +30% and spend >= INR 20,000)")
    print(f"  - Medium Risk:   {med_cnt:>5,} postings (Moderate spend variance: variance >= +15%)")
    print(f"  - Low Risk:      {low_cnt:>5,} postings (Controlled / on-forecast expenditure)")

    # Save top cost center spend forecast overrun watchlist CSV
    scored_df[scored_df['risk_category'].isin(['Critical Risk', 'High Risk'])].head(1500).to_csv(
        "reports/cost_center_spend_forecast_watchlist.csv", index=False
    )

    # 7. Compute 32-Feature Importance & Explainability
    print("\n>>> STEP 7: Computing 32-Feature Importance for Time-Series Spend Forecasting...")
    feat_imp = compute_feature_importance(forecaster.model, feature_cols, X_train_scaled, y_true=y_train)
    print("\n=== TOP 15 TIME-SERIES FEATURE IMPORTANCE SCORES ===")
    print(feat_imp.head(15).to_string(index=False))
    feat_imp.to_csv("reports/feature_importance.csv", index=False)

    # 8. Export Enriched 3-Sheet Excel Reports
    print("\n>>> STEP 8: Generating Executive 3-Sheet Enriched Excel Report...")
    desc_text = (
        f"Forecast departmental spending using Time-Series Machine Learning models (Gradient Boosting, "
        f"Random Forest, Bayesian AutoRegressive, Ridge, Holt-Winters EWMA) trained on {train_year} "
        f"historical data to forecast {forecast_year} departmental spend across SAP Cost Centers (RSEG/RBKP/EKKN), "
        f"Cost Center Master (CSKS), G/L Cost Elements (SKAT/SKA1), and Maintenance Orders (AUFK)."
    )
    excel_path = export_enriched_excel_report(
        df=scored_df,
        output_path=output_excel,
        insight_id="ML_13",
        use_case="Cost Center Spend Forecasting",
        description=desc_text,
        feature_importance_df=feat_imp
    )

    for extra_path in [
        "ML_13_Cost_Center_Spend_Forecasting_Enriched_Output.xlsx",
        os.path.join("reports", "ML_13_Cost_Center_Spend_Forecasting_Enriched_Output.xlsx"),
        os.path.join("reports", os.path.basename(output_excel))
    ]:
        if os.path.abspath(extra_path) != os.path.abspath(excel_path):
            os.makedirs(os.path.dirname(os.path.abspath(extra_path)), exist_ok=True)
            shutil.copy2(excel_path, extra_path)
            print(f"[SUCCESS] Copied Executive Enriched Excel Report to: {extra_path}")

    # 9. Persist Model Artifacts
    print("\n>>> STEP 9: Persisting Model Pipeline & Benchmark Artifacts...")
    save_single_model_pipeline(
        forecaster, scaler, feature_cols, suite=suite, comparison_df=comparison_df, output_dir="models"
    )

    print("\n" + "=" * 105)
    print("  TIME-SERIES COST CENTER SPEND FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  CHAMPION MODEL: {champion_model_name} (Accuracy: {champion_acc:.2f}%, R²: {champion_r2:.4f})")
    print(f"  EXCEL REPORT GENERATED: {os.path.abspath(excel_path)}")
    print("=" * 105)
    return excel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAP Cost Center Spend Forecasting (Time Series: 2025 -> 2026)")
    parser.add_argument("--output", type=str, default="ML_13_Cost_Center_Spend_Forecasting_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(output_excel=args.output)
