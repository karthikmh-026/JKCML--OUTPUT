"""
Evaluation and Comparison Module - Use Case 13: Cost Center Spend Forecasting (Time Series)
Calculates Time-Series forecasting evaluation metrics (MAE, RMSE, MAPE, Forecast Accuracy %,
R² Score, Forecast Bias %, and Correlation), ranks all 5 Time-Series ML models to select the #1
Champion Model, and computes 32-feature importance and mathematical explanations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_models(predictions_dict: dict, y_true: np.ndarray) -> pd.DataFrame:
    """
    Evaluates and ranks all 5 Time-Series Forecasting models on the 2025 test split
    against actual Cost Center departmental expenditure.
    """
    metrics = []

    safe_actual = np.maximum(100.0, np.abs(y_true))

    for name, preds in predictions_dict.items():
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)

        # Percentage absolute error capped per sample to avoid division-by-zero distortion
        abs_pct_errors = np.clip(np.abs(y_true - preds) / safe_actual, 0.0, 2.5)
        mape = float(np.mean(abs_pct_errors) * 100.0)

        # Forecast Accuracy % = max(0, 1 - MAPE) * 100
        accuracy_pct = float(np.clip(100.0 - mape, 0.0, 100.0))

        # Directional Forecast Bias % (Positive = underforecasting, Negative = overforecasting)
        bias_pct = float(np.mean((y_true - preds) / safe_actual) * 100.0)

        # Pearson Correlation
        if np.std(preds) > 1e-6 and np.std(y_true) > 1e-6:
            corr = float(np.corrcoef(y_true, preds)[0, 1])
        else:
            corr = 0.0

        metrics.append({
            'Model': name,
            'Accuracy': round(accuracy_pct / 100.0, 4),
            'Accuracy (%)': round(accuracy_pct, 2),
            'R² Score': round(r2, 4),
            'MAE (INR)': round(mae, 2),
            'RMSE (INR)': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Forecast Bias (%)': round(bias_pct, 2),
            'Correlation': round(corr, 4)
        })

    comparison_df = (
        pd.DataFrame(metrics)
        .sort_values(by=['Accuracy (%)', 'R² Score'], ascending=[False, False])
        .reset_index(drop=True)
    )
    comparison_df.insert(0, 'Rank', [f"#{i+1}" for i in range(len(comparison_df))])
    return comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: str = None):
    """
    Generates and saves a comparative performance bar chart for the 5 Time-Series forecasting models.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(13, 6))
    melted = pd.melt(
        comparison_df,
        id_vars=['Model'],
        value_vars=['Accuracy (%)', 'R² Score'],
        var_name='Metric',
        value_name='Score'
    )

    palette = sns.color_palette("Set2", 2)
    ax = sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette=palette)
    plt.title("5 Time-Series ML Models Performance & Accuracy Comparison - Cost Center Spend Forecasting (2025 -> 2026)", fontsize=12, fontweight='bold')
    plt.xlabel("Time-Series Machine Learning Model", fontsize=11)
    plt.ylabel("Score / Percentage", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    for p in ax.patches:
        height = p.get_height()
        if height > 0.04:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                textcoords='offset points'
            )

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved 5-model forecasting comparison chart to: {save_path}")
    plt.close()


def plot_roc_pr_curves(predictions_dict: dict, y_true: np.ndarray, output_dir: str = None):
    """
    Generates time-series actual vs forecasted scatter and residual distribution plots across models.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    champion_name = list(predictions_dict.keys())[0]
    champ_preds = predictions_dict[champion_name]

    sample_size = min(1500, len(y_true))
    idx = np.random.RandomState(42).choice(len(y_true), size=sample_size, replace=False)

    y_sample = y_true[idx]
    p_sample = champ_preds[idx]

    ax1.scatter(np.log1p(np.maximum(0, y_sample)), np.log1p(np.maximum(0, p_sample)), alpha=0.45, color='royalblue', edgecolors='none')
    max_val = max(np.log1p(np.maximum(0, y_sample)).max(), np.log1p(np.maximum(0, p_sample)).max())
    ax1.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Forecast (45° Line)')
    ax1.set_title(f"Actual vs Forecasted Spend (Log-Scale) - {champion_name}", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Log(Actual Spend INR)")
    ax1.set_ylabel("Log(Forecasted Spend INR)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for name, preds in list(predictions_dict.items())[:3]:
        residuals = np.log1p(np.maximum(0, preds[idx])) - np.log1p(np.maximum(0, y_sample))
        sns.kdeplot(residuals, ax=ax2, label=name, lw=2)

    ax2.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax2.set_title("Forecast Error Residual Distributions (Log Scale)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Forecast Residual (Log Pred - Log Actual)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "roc_pr_curves.png")
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved Time-Series actual vs forecast visual chart to: {save_path}")
    plt.close()


FEATURE_EXPLANATIONS = {
    'log_lag_1_month_spend_inr': '(Log-stabilized Cost Center departmental spend in the previous month t-1 capturing immediate baseline autoregressive momentum)',
    'log_lag_2_month_spend_inr': '(Log-stabilized Cost Center departmental spend 2 months prior t-2 capturing intermediate multi-period spending carryover)',
    'log_lag_3_month_spend_inr': '(Log-stabilized Cost Center departmental spend 3 months prior t-3 establishing quarterly expenditure momentum)',
    'log_rolling_mean_3m_spend_inr': '(Log-stabilized 3-month moving average spend smoothing short-term invoice posting fluctuations and establishing trend baseline)',
    'log_ewma_spend_inr': '(Log-stabilized Exponentially Weighted Moving Average EWMA(alpha=0.40) reflecting recency-weighted departmental burn rate)',
    'cc_mom_spend_growth_pct': '(Month-over-Month percentage spend growth rate measuring acceleration or deceleration in cost center expenditure)',
    'cc_seasonality_index': '(Ratio of monthly actual spend to 3-month rolling mean spend capturing seasonal operational peaks and budget cycles)',
    'cc_spend_trend_slope': '(Normalized time-series spend trend slope (%) measuring divergence between short-term EWMA and long-term moving average)',
    'rolling_std_3m_spend_inr': '(3-month rolling standard deviation of departmental spend measuring localized time-series expenditure volatility)',
    'rolling_max_3m_spend_inr': '(Peak expenditure recorded in the trailing 3-month rolling window establishing recent maximum expenditure ceiling)',
    'cc_monthly_tx_count': '(Total count of invoice line items posted to the Cost Center in the given month reflecting operational transaction density)',
    'cc_monthly_mean_tx_inr': '(Average invoice line item spend for the Cost Center in the given month measuring transaction ticket size)',
    'cc_monthly_max_tx_inr': '(Maximum single invoice posting amount for the Cost Center in the given month capturing large milestone expenses)',
    'cc_hist_mean_spend_inr': '(Long-term historical mean transaction spend across all observed periods for the Cost Center KOSTL)',
    'cc_hist_std_spend_inr': '(Long-term standard deviation of transaction spend within the Cost Center establishing historical baseline dispersion)',
    'dept_hist_mean_spend_inr': '(Cross-sectional mean transaction spend for the Departmental Spend Category across all peer Cost Centers)',
    'expected_spend_baseline_inr': '(Composite baseline expected spend blending Cost Center monthly mean, historical mean, and departmental peer mean)',
    'cc_spend_zscore': '(Standardized statistical Z-score of transaction spend relative to the Cost Center historical distribution)',
    'cc_spend_volatility_cv': '(Historical coefficient of variation std/mean of spend within the Cost Center quantifying intrinsic budget volatility)',
    'tx_share_of_monthly_spend': '(Proportional concentration [0-1] of the individual invoice item within the total monthly cost center spend)',
    'posting_month': '(Calendar posting month 1..12 capturing annual budgetary cycles, plant overhaul windows, and fiscal quarters)',
    'posting_quarter': '(Fiscal calendar quarter Q1..Q4 capturing quarterly financial closing and budget allocation cycles)',
    'posting_day_of_month': '(Day of the month 1..31 capturing end-of-month invoice processing spikes and milestone payment windows)',
    'posting_day_of_week': '(Day of the week 0..6 capturing weekday administrative invoice verification vs weekend shifts)',
    'is_weekend_posting': '(Binary indicator = 1 for weekend postings identifying non-standard off-cycle invoice verification entries)',
    'is_month_end_window': '(Binary indicator = 1 for postings on or after day 25 capturing month-end accounting cut-off rush)',
    'posting_lag_days': '(Absolute elapsed days between invoice document date BLDAT and posting date BUDAT capturing entry latency)',
    'month_sin': '(Sine Fourier cyclical seasonal encoding sin(2*pi*month/12) ensuring smooth cyclical continuity across year transitions)',
    'month_cos': '(Cosine Fourier cyclical seasonal encoding cos(2*pi*month/12) capturing seasonal harmonic phases in cement operations)',
    'log_po_commitment_inr': '(Log-stabilized open Purchase Order commitment pipeline assigned to the Cost Center in EKKN)',
    'po_commitment_to_spend_ratio': '(Ratio of open PO commitments to quarterly spend measuring forward encumbered budget coverage)',
    'cc_pm_order_count': '(Count of active SAP Plant Maintenance and Internal Orders AUFK assigned to the Cost Center)'
}


def compute_feature_importance(champion_model, feature_names: list, X_scaled: np.ndarray, y_true: np.ndarray = None):
    """
    Computes natural tree-based feature importance for Time-Series Cost Center Spend Forecasting across all 32 features.
    """
    if hasattr(champion_model, 'feature_importances_'):
        raw_importances = champion_model.feature_importances_
    elif hasattr(champion_model, 'model') and hasattr(champion_model.model, 'feature_importances_'):
        raw_importances = champion_model.model.feature_importances_
    else:
        from sklearn.ensemble import RandomForestRegressor
        surrogate = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        surrogate.fit(X_scaled, y_true)
        raw_importances = surrogate.feature_importances_

    sort_indices = np.argsort(-raw_importances)
    sorted_features = [feature_names[i] for i in sort_indices]
    sorted_imps = raw_importances[sort_indices]

    explanations = [
        FEATURE_EXPLANATIONS.get(feat, f"(Time-Series Cost Center Spend Forecasting feature for {feat})")
        for feat in sorted_features
    ]

    importance_df = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_imps,
        'Direct Mathematical Influence On Cost Center Spend Forecasting & Budget Governance:': explanations
    })

    return importance_df
