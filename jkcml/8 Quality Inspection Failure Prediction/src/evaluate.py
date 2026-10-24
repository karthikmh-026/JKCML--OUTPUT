"""
Evaluation and Ranking Module - Use Case 8: Quality Inspection Failure Prediction
Evaluates the 5 Unsupervised Machine Learning Models on held-out test split,
computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Specificity,
ranks all 5 models to select the #1 Champion Model, and computes 32-feature
importance and visual evaluation figures.
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


def evaluate_models(predictions_dict: dict, y_true: np.ndarray) -> pd.DataFrame:
    """
    Evaluates and ranks all 5 unsupervised models on held-out test set against
    post-hoc audit benchmark inspection failure labels.
    """
    metrics = []

    for name, res in predictions_dict.items():
        preds = res['preds']
        scores = res['scores']

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
    plt.title("5 Unsupervised ML Models Performance & Accuracy Comparison - Quality Inspection Failure Prediction", fontsize=13, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.xlabel("Unsupervised Machine Learning Model", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

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
        print(f"[INFO] Saved 5-model comparison chart to: {save_path}")
    plt.close()


def plot_roc_pr_curves(predictions_dict: dict, y_true: np.ndarray, output_dir: str = None):
    """Plots ROC and Precision-Recall Curves for all 5 unsupervised models."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, res in predictions_dict.items():
        scores = res['scores']
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = roc_auc_score(y_true, scores)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)

        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        ax2.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)

    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline')
    ax1.set_title("ROC Curves - Quality Inspection Failure Prediction (5 Models)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Precision-Recall Curves - Quality Inspection Failure Prediction", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "roc_pr_curves.png")
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved ROC/PR curves plot to: {save_path}")
    plt.close()


def plot_failure_risk_distribution(scored_df: pd.DataFrame, output_dir: str = None):
    """Plots the distribution of inspection lots across the 4 Quality Risk Tiers."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tier_order = ["Critical Failure Risk", "High Risk", "Medium Risk", "Low Risk"]
    tier_counts = scored_df['risk_category'].value_counts().reindex(tier_order).fillna(0)

    plt.figure(figsize=(10, 5))
    colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c']
    bars = plt.bar(tier_counts.index, tier_counts.values, color=colors, edgecolor='black', alpha=0.85)

    plt.title("Distribution of Inspection Lots across Quality Failure Risk Tiers", fontsize=13, fontweight='bold')
    plt.xlabel("Quality Risk Tier", fontsize=11)
    plt.ylabel("Number of Inspection Lots", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    total = len(scored_df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / max(1, total)) * 100.0
        plt.text(
            bar.get_x() + bar.get_width() / 2., height + 5,
            f"{int(height):,}\n({pct:.1f}%)",
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "failure_risk_distribution.png")
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved failure risk distribution plot to: {save_path}")
    plt.close()


FEATURE_EXPLANATIONS = {
    'non_conforming_ratio': '(Combined proportion of inspection lot posted to scrap or blocked inventory)',
    'scrap_qty_num': '(Total physical material units rejected and posted directly to scrap LMENGE02)',
    'scrap_ratio': '(Ratio of scrapped material units to total lot inspection quantity)',
    'blocked_qty_num': '(Quantity sequestered and transferred to blocked/quarantined stock LMENGE04)',
    'blocked_ratio': '(Ratio of quarantined/blocked stock to total lot inspection quantity)',
    'destroyed_sample_qty_num': '(Quantity consumed or irreversibly destroyed during destructive lab testing LMENGEPR)',
    'destroyed_sample_ratio': '(Proportion of sample quantity destroyed during testing)',
    'sample_to_lot_ratio': '(Sampling inspection intensity: sample quantity divided by total lot size)',
    'inspection_lead_days': '(Total elapsed days between inspection lot creation and planned finish)',
    'ud_lag_days': '(Processing delay in days between lot creation and usage decision sign-off)',
    'is_overdue_inspection': '(Indicator if inspection lot is unresolved past its scheduled end date)',
    'log_lot_financial_value': '(Log-scaled financial value exposed in lot: Lot Quantity * Valuation Unit Price)',
    'log_unit_price': '(Log-stabilized unit valuation price from SAP Material Valuation MBEW)',
    'log_lot_quantity': '(Log-stabilized total inspection lot quantity QALS-LOSMENGE)',
    'log_sample_quantity': '(Log-stabilized planned sample size QALS-LMENGEZUB)',
    'unrestricted_ratio': '(Proportion of lot accepted and released into unrestricted stock LMENGE01)',
    'vendor_is_rare': '(1 if vendor has 2 or fewer historical deliveries, indicating new/unvetted supplier)',
    'log_vendor_frequency': '(Log of historical inspection lot deliveries received from this vendor)',
    'log_material_frequency': '(Log of total historical inspection lots processed for this material number)',
    'log_plant_frequency': '(Log of total inspection volume handled at this manufacturing plant)',
    'planned_delivery_days_num': '(Planned delivery lead time days configured in Material Master MARC-PLIFZ)',
    'log_safety_stock': '(Log-scaled safety stock level configured at plant level MARC-EISBE)',
    'log_net_weight': '(Log of net unit weight from Material Master MARA-NTGEW)',
    'insp_type_code': '(Numeric code representing inspection type: 01 Goods Receipt, 03 In-Process, 08 Transfer)',
    'origin_code': '(Inspection lot origin code QALS-HERKUNFT)',
    'material_type_code': '(Material type classification: Raw Materials, Packaging, Semi-Finished, Finished)',
    'has_batch': '(1 if inspection lot is assigned a traceable batch number CHARG)',
    'has_po': '(1 if inspection lot originated from a formal Purchase Order EBELN)',
    'status_sample_flag': '(1 if inspection sample was formally drawn and recorded in SAP QALS-STAT02)',
    'is_monthend_rush': '(1 if inspection lot was created during period-end rush days >= 25th)',
    'is_weekend_creation': '(1 if inspection lot was created on Saturday or Sunday)',
    'scheduled_duration_days': '(Duration of planned inspection window: scheduled finish minus scheduled start)'
}


def compute_feature_importance(champion_model, feature_names: list, X_scaled: np.ndarray, y_true: np.ndarray = None) -> pd.DataFrame:
    """
    Computes natural tree-based feature importance for Quality Inspection Failure Prediction.
    When y_true is provided, trains a Random Forest ensemble on benchmark exceptions.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # Subsample if large to ensure fast execution
    if len(X_scaled) > 50000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_scaled), size=50000, replace=False)
        X_sub = X_scaled[idx]
        y_sub = y_true[idx] if y_true is not None else None
    else:
        X_sub = X_scaled
        y_sub = y_true

    if y_sub is not None and len(np.unique(y_sub)) > 1:
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        rf.fit(X_sub, y_sub)
        raw_importances = rf.feature_importances_
    else:
        if hasattr(champion_model, 'score_samples'):
            scores = -champion_model.score_samples(X_sub)
        elif hasattr(champion_model, 'decision_function'):
            scores = -champion_model.decision_function(X_sub)
        else:
            scores = np.zeros(len(X_sub))
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        rf.fit(X_sub, scores)
        raw_importances = rf.feature_importances_

    sort_indices = np.argsort(-raw_importances)
    sorted_features = [feature_names[i] for i in sort_indices]
    sorted_imps = raw_importances[sort_indices]

    explanations = [
        FEATURE_EXPLANATIONS.get(feat, f"(Statistical quality failure feature: {feat})")
        for feat in sorted_features
    ]

    total_imp = np.sum(sorted_imps) if np.sum(sorted_imps) > 0 else 1.0
    relative_pct = (sorted_imps / total_imp) * 100.0

    fi_df = pd.DataFrame({
        'Rank': [f"#{i+1}" for i in range(len(sorted_features))],
        'Feature': sorted_features,
        'Importance': np.round(sorted_imps, 4),
        'Relative_Weight_Pct': np.round(relative_pct, 2),
        'Domain Impact on Inspection Failure Risk': explanations
    })
    return fi_df


def plot_feature_importance(fi_df: pd.DataFrame, output_dir: str = None, top_n: int = 15):
    """Plots top N most influential features driving quality inspection failure risk."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    top_df = fi_df.head(top_n).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(12, 7))
    palette = sns.color_palette("Blues_r", len(top_df))
    plt.barh(top_df['Feature'], top_df['Importance'], color=palette, edgecolor='navy', alpha=0.85)

    plt.title(f"Top {top_n} Most Influential Features Driving Quality Inspection Failure Risk", fontsize=13, fontweight='bold')
    plt.xlabel("Relative Feature Importance Score (Gini Impurity Reduction)", fontsize=11)
    plt.ylabel("Inspection & Material Features", fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    for i, v in enumerate(top_df['Importance']):
        plt.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved feature importance chart to: {save_path}")
    plt.close()
