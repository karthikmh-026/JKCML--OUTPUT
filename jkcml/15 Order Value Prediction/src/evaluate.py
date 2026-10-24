"""
Evaluation and Ranking Module - Use Case 15: Order Value Prediction
Category: Regression
Objective: Predict next order value of customers.

Evaluates 5 Supervised Regression Models on held-out test data:
- R2 Score (Coefficient of Determination)
- MAE (Mean Absolute Error in INR)
- RMSE (Root Mean Squared Error in INR)
- MAPE (Mean Absolute Percentage Error %)
- Accuracy within ±20% Tolerance (% of predictions within 20% of actual)
- Pearson Correlation (r)
- Feature Importance extraction and regression diagnostic visualizations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr


MODEL_DISPLAY_NAMES = {
    'random_forest': 'Random Forest Regressor',
    'gradient_boosting': 'Gradient Boosting Regressor',
    'hist_gradient_boosting': 'HistGradientBoosting Regressor',
    'decision_tree': 'Decision Tree Regressor',
    'ridge': 'Ridge Regressor'
}


def evaluate_models(predictions_dict: dict, y_true: np.ndarray) -> pd.DataFrame:
    """
    Evaluates all 5 regression models on the held-out test split,
    calculates key regression metrics, and ranks models primarily by R2 Score.
    """
    rows = []
    for key, preds in predictions_dict.items():
        preds_safe = np.maximum(0.0, preds)

        r2 = r2_score(y_true, preds_safe)
        mae = mean_absolute_error(y_true, preds_safe)
        rmse = root_mean_squared_error(y_true, preds_safe)

        # Median Absolute Percentage Error (robust against zero-division)
        rel_errors = np.abs(y_true - preds_safe) / np.maximum(1.0, y_true)
        mape = float(np.median(rel_errors) * 100.0)

        # Accuracy within ±20% tolerance band
        acc_20 = float(np.mean(rel_errors <= 0.20) * 100.0)

        corr, _ = pearsonr(y_true, preds_safe)

        rows.append({
            'Model': MODEL_DISPLAY_NAMES.get(key, key),
            'model_key': key,
            'R2 Score': round(float(r2), 4),
            'MAE (INR)': round(float(mae), 2),
            'RMSE (INR)': round(float(rmse), 2),
            'MAPE (%)': round(float(mape), 2),
            'Accuracy (±20%)': round(float(acc_20), 2),
            'Pearson r': round(float(corr), 4)
        })

    df_eval = pd.DataFrame(rows)
    # Rank descending by R2 Score, then lowest MAE
    df_eval = df_eval.sort_values(by=['R2 Score', 'MAE (INR)'], ascending=[False, True]).reset_index(drop=True)
    df_eval.insert(0, 'Rank', [f"#{i+1}" for i in range(len(df_eval))])
    return df_eval


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: str = "reports/figures"):
    """Generates a multi-metric comparison chart of the 5 regression models."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Normalized comparison: R2, Pearson r, and Accuracy (±20%)/100
    df_plot = comparison_df.copy()
    df_plot['Accuracy (Ratio)'] = df_plot['Accuracy (±20%)'] / 100.0

    melted = df_plot.melt(
        id_vars=['Model'],
        value_vars=['R2 Score', 'Pearson r', 'Accuracy (Ratio)'],
        var_name='Metric',
        value_name='Score'
    )

    palette = sns.color_palette("mako", len(comparison_df))
    ax = sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette=palette)
    plt.title("Use Case 15: Order Value Prediction - 5 Regression Models Ranking & Comparison", fontsize=14, weight='bold', pad=15)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score / Correlation", fontsize=12)
    plt.xlabel("Regression Metric", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "model_comparison_metrics.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved regression model comparison chart to {out_path}")


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_dir: str = "reports/figures"):
    """Scatter plot of Actual vs Predicted Next Order Value with y = x reference line."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 7))
    sample_size = min(3000, len(y_true))
    idx = np.random.choice(len(y_true), size=sample_size, replace=False)

    y_t_samp = y_true[idx] / 1e5  # in Lakhs
    y_p_samp = y_pred[idx] / 1e5

    plt.scatter(y_t_samp, y_p_samp, alpha=0.4, color='#1F4E79', edgecolors='none', s=25)
    max_val = max(np.percentile(y_t_samp, 99), np.percentile(y_p_samp, 99))
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction (y = x)')

    plt.title(f"Actual vs Predicted Next Order Value - #1 Champion ({model_name})", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Actual Next Order Value (INR Lakhs)", fontsize=11)
    plt.ylabel("Predicted Next Order Value (INR Lakhs)", fontsize=11)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Actual vs Predicted chart to {out_path}")


def plot_residuals_distribution(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_dir: str = "reports/figures"):
    """Histogram and KDE distribution of prediction residuals."""
    os.makedirs(output_dir, exist_ok=True)

    residuals = (y_pred - y_true) / 1e5  # in Lakhs
    residuals_clip = np.clip(residuals, np.percentile(residuals, 2), np.percentile(residuals, 98))

    plt.figure(figsize=(9, 5))
    sns.histplot(residuals_clip, bins=50, kde=True, color='#2F5597')
    plt.axvline(0, color='red', linestyle='--', lw=1.5, label='Zero Error')
    plt.title(f"Residual Error Distribution - #1 Champion ({model_name})", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Prediction Error: (Predicted - Actual) in INR Lakhs", fontsize=11)
    plt.ylabel("Order Frequency", fontsize=11)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "residuals_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved residuals distribution chart to {out_path}")


def compute_feature_importance(model, feature_cols: list) -> pd.DataFrame:
    """Computes feature importance ranking from tree-based regressor."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # Fallback for linear models
        importances = np.abs(getattr(model, 'coef_', np.ones(len(feature_cols))))

    importances = np.array(importances, dtype=float)
    if np.sum(importances) > 0:
        importances = importances / np.sum(importances)

    df_fi = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return df_fi


def plot_feature_importance(feat_imp_df: pd.DataFrame, output_dir: str = "reports/figures"):
    """Plots top 15 features driving next order value prediction."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    top15 = feat_imp_df.head(15)
    sns.barplot(data=top15, x='Importance', y='Feature', palette='crest')
    plt.title("Top 15 Features Driving Next Order Value Prediction (Champion Model)", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Normalized Feature Importance Score", fontsize=11)
    plt.ylabel("Engineered Feature", fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved feature importance chart to {out_path}")
