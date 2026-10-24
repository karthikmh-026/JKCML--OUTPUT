"""
Evaluation and Comparison Module - Use Case 7: Production Order Delay Prediction
Calculates performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC,
Silhouette Score, and Specificity), ranks all 5 Unsupervised ML Models
to select the #1 Ranked Champion Model, and generates visual diagnostic charts and feature importance.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier


def evaluate_models(predictions_dict: dict, y_true: np.ndarray) -> pd.DataFrame:
    """
    Evaluates and ranks all 5 Unsupervised ML Models on held-out test data
    against post-hoc SAP production delay ground-truth benchmarks.
    """
    metrics = []

    for name, res in predictions_dict.items():
        preds = res['preds']
        scores = res['scores']
        sil = res.get('silhouette', 0.0)

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = 0.5

        try:
            pr_auc = average_precision_score(y_true, scores)
        except Exception:
            pr_auc = 0.0

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4),
            'PR-AUC': round(pr_auc, 4),
            'Silhouette': round(sil, 4),
            'Specificity': round(specificity, 4),
            'True Positives': int(tp),
            'False Positives': int(fp),
            'False Negatives': int(fn),
            'True Negatives': int(tn)
        })

    comparison_df = (
        pd.DataFrame(metrics)
        .sort_values(by=['F1-Score', 'ROC-AUC', 'Accuracy'], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    comparison_df.insert(0, 'Rank', [f"#{i+1}" for i in range(len(comparison_df))])
    return comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: str = None):
    """Generates and saves a comparative performance bar chart for the 5 unsupervised models."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(13, 6))
    melted = pd.melt(
        comparison_df,
        id_vars=['Model'],
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        var_name='Metric',
        value_name='Score'
    )

    palette = sns.color_palette("Set2", 5)
    ax = sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette=palette)
    plt.title("5 Unsupervised ML Models Performance & Accuracy Comparison - Production Order Delay Prediction", fontsize=13, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.xlabel("Unsupervised Model", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0.05:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    if output_dir:
        out_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[REPORT] Saved model comparison chart to {out_path}")


def plot_roc_pr_curves(predictions_dict: dict, y_true: np.ndarray, output_dir: str = None):
    """Plots and saves ROC and Precision-Recall curves for all 5 models."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for name, res in predictions_dict.items():
        scores = res['scores']
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = roc_auc_score(y_true, scores)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc_val = average_precision_score(y_true, scores)
        axes[1].plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc_val:.3f})")

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc="lower right")

    axes[1].set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc="lower left")

    plt.suptitle("Discriminative Power of 5 Unsupervised Models (Production Order Delay)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_dir:
        out_path = os.path.join(output_dir, "roc_pr_curves.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[REPORT] Saved ROC & PR curves to {out_path}")


def plot_delay_risk_distribution(scored_df: pd.DataFrame, output_dir: str = None):
    """Plots the distribution of production orders across the 4 Delay Risk Tiers."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    tier_order = [
        "Tier 0: On-Time / Expedited Schedule",
        "Tier 1: Low Delay Risk / Minor Variance",
        "Tier 2: Moderate Delay Risk / Bottlenecked",
        "Tier 3: Critical Delay Exception / Past Due"
    ]
    tier_counts = scored_df['delay_risk_tier'].value_counts().reindex(tier_order, fill_value=0)

    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    axes[0].bar(range(len(tier_counts)), tier_counts.values, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_xticks(range(len(tier_counts)))
    axes[0].set_xticklabels(['Tier 0: On-Time', 'Tier 1: Low Risk', 'Tier 2: Moderate', 'Tier 3: Critical'], rotation=15, fontsize=10)
    axes[0].set_title("Production Order Volume by Delay Risk Tier", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Number of Production Orders", fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for i, v in enumerate(tier_counts.values):
        pct = (v / len(scored_df)) * 100.0
        axes[0].text(i, v + (max(tier_counts.values) * 0.01), f"{v:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Distribution of continuous delay probability
    sns.histplot(scored_df['predicted_delay_probability'], bins=30, kde=True, ax=axes[1], color='#1f77b4', edgecolor='black')
    axes[1].axvline(0.50, color='red', linestyle='--', lw=2, label='Delay Threshold (0.50)')
    axes[1].set_title("Continuous P(Delay) Probability Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Probability of Production Delay P(Delay)", fontsize=10)
    axes[1].set_ylabel("Order Count", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    plt.suptitle("Production Order Delay Risk Segmentation (#1 Champion Model)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_dir:
        out_path = os.path.join(output_dir, "delay_risk_distribution.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[REPORT] Saved delay risk distribution plot to {out_path}")


def compute_feature_importance(X_train_scaled: np.ndarray, y_train: np.ndarray, feature_cols: list) -> pd.DataFrame:
    """
    Computes surrogate Gini feature importance using a fast Random Forest classifier
    to explain key operational drivers behind production order delays.
    """
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': np.round(importances, 6)
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    fi_df.insert(0, 'Rank', range(1, len(fi_df) + 1))
    fi_df['Relative_Weight_Pct'] = np.round((fi_df['Importance'] / fi_df['Importance'].sum()) * 100.0, 2)
    fi_df['Cumulative_Pct'] = np.round(fi_df['Relative_Weight_Pct'].cumsum(), 2)

    return fi_df


def plot_feature_importance(fi_df: pd.DataFrame, output_dir: str = None, top_n: int = 15):
    """Plots and saves the top N most important features explaining delay probability."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(11, 7))
    top_fi = fi_df.head(top_n).sort_values(by='Importance', ascending=True)

    colors = sns.color_palette("viridis", len(top_fi))
    bars = plt.barh(top_fi['Feature'], top_fi['Importance'], color=colors, edgecolor='black', alpha=0.85)

    plt.title(f"Top {top_n} Most Influential Delay Predictors - Production Order Operations", fontsize=13, fontweight='bold')
    plt.xlabel("Surrogate Gini Feature Importance", fontsize=11)
    plt.ylabel("Operational Feature", fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    for bar, pct in zip(bars, top_fi['Relative_Weight_Pct']):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%",
                 va='center', ha='left', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if output_dir:
        out_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[REPORT] Saved feature importance plot to {out_path}")
