"""
Evaluation and Comparison Module - Use Case 21: Material Master Data Quality Scoring
Calculates evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Specificity),
ranks the 5 unsupervised anomaly detection models to select the #1 Champion Model,
and computes 32-feature importance and explainability.
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
    Evaluates and ranks all 5 unsupervised models on the test set against post-hoc
    audit benchmark material master quality exception labels.
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
    plt.title("5 Unsupervised ML Models Performance & Accuracy Comparison - Material Master Data Quality Scoring", fontsize=13, fontweight='bold')
    plt.ylim(0, 1.12)
    plt.xlabel("Unsupervised Anomaly Detection Model", fontsize=11)
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
    """
    Plots ROC and Precision-Recall Curves for all 5 unsupervised models.
    """
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
    ax1.set_title("ROC Curves - Material Master Data Quality Scoring (5 Models)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Precision-Recall Curves - Material Master Data Quality Scoring", fontsize=13, fontweight='bold')
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


FEATURE_EXPLANATIONS = {
    'mdq_defect_index': '(Composite multi-view Material Master Data Quality Defect Score [0-100] across MARA, MARC, MBEW & MAKT)',
    'mdq_health_score': '(Overall Material Master Data Quality Health Score [0-100], defined as 100 - mdq_defect_index)',
    'missing_critical_fields_cnt': '(Total count of unpopulated mandatory governance attributes across Basic, MRP, Purchasing & Accounting views)',
    'completeness_ratio': '(Proportion of 12 core SAP Material Master governance attributes properly populated in [0.0, 1.0])',
    'is_missing_valuation_view': '(1 if Valuated Material Type lacks Accounting Valuation Class MBEW-BKLAS or Price Control MBEW-VPRSV)',
    'is_zero_valuation_price': '(1 if Valuated Material carries zero Moving Average Price VERPR and zero Standard Price STPRS)',
    'is_price_control_mismatch': '(1 if Price Control S has zero STPRS or Price Control V has zero VERPR on a valuated material)',
    'is_negative_stock_or_value': '(1 if Total Valuated Stock MBEW-LBKUM < 0 or Total Stock Value MBEW-SALK3 < 0)',
    'is_stock_value_inconsistent': '(1 if Positive Stock LBKUM > 0 has zero value SALK3 <= 0, or zero stock carries residual value)',
    'is_missing_purchasing_group': '(1 if Externally Procured / Spare / Raw Material is missing Purchasing Group MARC-EKGRP)',
    'is_mrp_governance_incomplete': '(1 if MRP Type MARC-DISMM is blank, or active MRP Type lacks MRP Controller DISPO / Procurement Type BESKZ)',
    'is_missing_profit_center': '(1 if Plant Material record MARC-PRCTR lacks Profit Center assignment for CO-PA / segment reporting)',
    'is_missing_weight_spec': '(1 if Physical Goods Material is missing Gross Weight MARA-BRGEW, Net Weight MARA-NTGEW, or Weight Unit GEWEI)',
    'is_weight_inconsistent': '(1 if Net Weight exceeds Gross Weight NTGEW > BRGEW, or one weight is populated while the other is zero)',
    'is_duplicate_description': '(1 if identical normalized Material Description MAKT-MAKTX is shared across >=4 distinct MATNR codes)',
    'dup_description_count': '(Total count of distinct MATNR codes sharing the identical normalized Material Description within Type & Group)',
    'is_cryptic_or_short_desc': '(1 if Material Description MAKT-MAKTX is <7 characters, single-token, or placeholder text)',
    'desc_char_length': '(Character length of Material Description MAKT-MAKTX measuring descriptive specificity)',
    'desc_token_count': '(Word token count in Material Description MAKT-MAKTX)',
    'is_missing_old_mat_no': '(1 if Legacy / Cross-Reference Old Material Number MARA-BISMT is blank)',
    'is_missing_division': '(1 if Material Master Division MARA-SPART is unpopulated)',
    'is_missing_prod_hierarchy': '(1 if Commercial / Finished Product lacks Product Hierarchy MARA-PRDHA)',
    'mara_views_maintained_cnt': '(Count of active functional maintenance status views configured in MARA-PSTAT)',
    'marc_views_maintained_cnt': '(Count of active plant-specific maintenance status views configured in MARC-PSTAT)',
    'view_sync_gap': '(Synchronization gap between enterprise views MARA-VPSTA and plant views MARC-PSTAT)',
    'effective_unit_price': '(Active accounting unit valuation price in INR: STPRS/PEINH for S-price or VERPR/PEINH for V-price)',
    'log_effective_price': '(Log-stabilized active unit valuation price ln(1 + effective_unit_price))',
    'mat_group_price_zscore': '(Standard deviations of material log valuation price from its Material Group MATKL peer mean)',
    'implied_vs_book_price_diff_pct': '(Percentage discrepancy between implied stock unit value SALK3/LBKUM and book valuation price)',
    'log_valuated_stock': '(Log-stabilized total valuated inventory quantity ln(1 + |MBEW-LBKUM|))',
    'log_stock_value': '(Log-stabilized total valuated inventory value ln(1 + |MBEW-SALK3|))',
    'days_since_last_change': '(Elapsed days between Material Creation Date MARA-ERSDA and Last Change Date MARA-LAEDA)'
}


def compute_feature_importance(isolation_forest_model, feature_names: list, X_scaled: np.ndarray, y_true: np.ndarray = None):
    """
    Computes natural tree-based feature importance for Material Master Data Quality Scoring across all 32 features.
    When y_true is provided, trains a Random Forest ensemble on the benchmark quality exceptions,
    yielding genuine domain importances.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # Subsample if dataset is massive to ensure fast training
    max_feat_samples = 30000
    if len(X_scaled) > max_feat_samples:
        np.random.seed(42)
        idx = np.random.choice(len(X_scaled), size=max_feat_samples, replace=False)
        X_sub = X_scaled[idx]
        y_sub = y_true[idx] if y_true is not None else None
    else:
        X_sub = X_scaled
        y_sub = y_true

    if y_sub is not None and len(np.unique(y_sub)) > 1:
        model = RandomForestClassifier(n_estimators=100, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        model.fit(X_sub, y_sub)
        raw_importances = model.feature_importances_
    else:
        anomaly_scores = -isolation_forest_model.decision_function(X_sub)
        model = RandomForestRegressor(n_estimators=100, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        model.fit(X_sub, anomaly_scores)
        raw_importances = model.feature_importances_

    sort_indices = np.argsort(-raw_importances)
    sorted_features = [feature_names[i] for i in sort_indices]
    sorted_imps = raw_importances[sort_indices]

    explanations = [
        FEATURE_EXPLANATIONS.get(feat, f"(Statistical material data quality feature for {feat})")
        for feat in sorted_features
    ]

    importance_df = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_imps,
        'Direct Mathematical Influence On Material Master Data Quality Risk:': explanations
    })

    return importance_df
