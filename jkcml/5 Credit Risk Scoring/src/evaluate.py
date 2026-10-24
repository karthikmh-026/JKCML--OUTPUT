"""
Evaluation and Comparison Module - Use Case 5: Credit Risk Scoring
Calculates performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC,
Silhouette Score, and Specificity), ranks all 5 Unsupervised Clustering ML models
to select the #1 Ranked Champion Model, and generates visual charts and feature importance.
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
    Evaluates and ranks all 5 Unsupervised Clustering models on the held-out test set
    against post-hoc benchmark Basel II / IFRS 9 default indicators plus Silhouette Score.
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
    """
    Generates and saves a comparative performance bar chart for the 5 unsupervised models.
    """
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
    plt.title("5 Unsupervised ML Models Performance & Accuracy Comparison - Customer Credit Risk Scoring", fontsize=13, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.xlabel("Unsupervised Model", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(save_path, dpi=300)
        print(f"[REPORT] Saved model comparison chart to: {save_path}")
    plt.close()


def plot_roc_pr_curves(predictions_dict: dict, y_true: np.ndarray, output_dir: str = None):
    """
    Plots multi-model ROC and Precision-Recall curves.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (name, res) in enumerate(predictions_dict.items()):
        scores = res['scores']
        c = colors[idx % len(colors)]

        # ROC Curve
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=c, lw=2)
        except Exception:
            pass

        # PR Curve
        try:
            prec, rec, _ = precision_recall_curve(y_true, scores)
            pr_auc = average_precision_score(y_true, scores)
            axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=c, lw=2)
        except Exception:
            pass

    # Baseline lines
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Chance')
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='lower right', fontsize=9)

    pos_rate = np.mean(y_true)
    axes[1].axhline(y=pos_rate, color='k', linestyle='--', lw=1.5, alpha=0.7, label=f'Base Delinquency Rate ({pos_rate:.1%})')
    axes[1].set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall (Coverage)")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "roc_pr_curves.png")
        plt.savefig(save_path, dpi=300)
        print(f"[REPORT] Saved ROC & PR curves to: {save_path}")
    plt.close()


def plot_credit_risk_distribution(df_scored: pd.DataFrame, output_dir: str = None):
    """
    Plots credit risk tier breakdown and probability distribution.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Tier count bar chart
    tier_counts = df_scored['credit_risk_tier_name'].value_counts()
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    axes[0].bar(tier_counts.index, tier_counts.values, color=colors[:len(tier_counts)], edgecolor='black', alpha=0.85)
    axes[0].set_title("Customer Credit Risk Tier Segmentation Distribution", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Number of Customers")
    axes[0].tick_params(axis='x', rotation=25)
    for i, v in enumerate(tier_counts.values):
        axes[0].text(i, v + max(tier_counts.values) * 0.02, f"{v:,}", ha='center', fontweight='bold', fontsize=9)

    # 2. Probability of Default histogram
    sns.histplot(df_scored['probability_of_default'], bins=30, kde=True, ax=axes[1], color='#3366cc')
    axes[1].axvline(0.50, color='red', linestyle='--', lw=2, label='High Risk Threshold (0.50)')
    axes[1].set_title("Calibrated Probability of Customer Default P(Default)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("P(Default)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "cluster_distribution.png")
        plt.savefig(save_path, dpi=300)
        print(f"[REPORT] Saved cluster distribution chart to: {save_path}")
    plt.close()


def compute_feature_importance(X_train_scaled: np.ndarray, y_train: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Surrogate tree explainability model to compute feature importance rankings for Credit Risk.
    """
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    importance = rf.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance,
        'Relative_Weight_Pct': np.round((importance / importance.sum()) * 100.0, 2)
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    fi_df.insert(0, 'Rank', [f"#{i+1}" for i in range(len(fi_df))])
    return fi_df


def plot_feature_importance(fi_df: pd.DataFrame, output_dir: str = None, top_n: int = 15):
    """
    Generates and saves a horizontal bar chart of the top N most important features.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    top_df = fi_df.head(top_n).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(11, 7))
    palette = sns.color_palette("mako", len(top_df))
    bars = plt.barh(top_df['Feature'], top_df['Importance'], color=palette, edgecolor='black', alpha=0.85)

    plt.title(f"Top {top_n} Most Influential Features Driving Customer Default Probability", fontsize=13, fontweight='bold')
    plt.xlabel("Surrogate Model Feature Importance (Gini Impurity Reduction)", fontsize=11)
    plt.ylabel("Feature Name", fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    for bar, pct in zip(bars, top_df['Relative_Weight_Pct']):
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%",
                 ha='left', va='center', fontsize=9, fontweight='bold', color='#1f497d')

    plt.xlim(0, max(top_df['Importance']) * 1.18)
    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(save_path, dpi=300)
        print(f"[REPORT] Saved feature importance chart to: {save_path}")
    plt.close()

