"""
Evaluation and Comparison Module - Use Case 26: Breakdown Root Cause Analysis (NLP)
Calculates evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Specificity,
and Silhouette Score), ranks the 5 Unsupervised NLP & Clustering models to select the #1 Champion Model,
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
    Evaluates and ranks all 5 Unsupervised NLP & Clustering models on the test set against post-hoc
    benchmark Breakdown Root Cause exception labels, plus unsupervised Silhouette Score.
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
        .sort_values(by=['ROC-AUC', 'PR-AUC', 'F1-Score', 'Accuracy'], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )
    comparison_df.insert(0, 'Rank', [f"#{i+1}" for i in range(len(comparison_df))])
    return comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: str = None):
    """
    Generates and saves a comparative performance bar chart for the 5 unsupervised NLP clustering models.
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
    plt.title("5 Unsupervised NLP Clustering ML Models Performance & Accuracy Comparison - Breakdown Root Cause Analysis", fontsize=12, fontweight='bold')
    plt.ylim(0, 1.12)
    plt.xlabel("Unsupervised NLP Clustering Model", fontsize=11)
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
        print(f"[INFO] Saved 5-model clustering comparison chart to: {save_path}")
    plt.close()


def plot_roc_pr_curves(predictions_dict: dict, y_true: np.ndarray, output_dir: str = None):
    """
    Plots ROC and Precision-Recall Curves for all 5 unsupervised NLP clustering models.
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
    ax1.set_title("ROC Curves - Breakdown Root Cause Analysis (5 Unsupervised Models)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Precision-Recall Curves - Breakdown Root Cause Analysis", fontsize=12, fontweight='bold')
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
    'nlp_rca_confidence_score': '(Composite NLP Breakdown Root Cause Extraction Confidence Score [0-100] combining Diagnostic Clarity 34%, Root-Cause Specificity 36%, and PM Execution Alignment 30%)',
    'breakdown_severity_risk_index': '(Composite Breakdown & Unverified Root-Cause Severity Risk Index [0-100], defined as 100 - nlp_rca_confidence_score)',
    'text_diagnostic_clarity_score': '(Normalized Maintenance Text Diagnostic Clarity Score [0-100] from AUFK-KTEXT and AFRU-LTXA1 length, word density, and equipment tag specificity)',
    'rca_semantic_specificity_score': '(Normalized NLP Root-Cause Semantic Specificity Score [0-100] measuring dominant failure-mode affinity and category margin)',
    'pm_execution_alignment_score': '(Normalized SAP Plant Maintenance Execution & Work-Center Alignment Score [0-100] measuring TECO completion, confirmation logging, and VAPLZ discipline match)',
    'nlp_mechanical_bearing_drive_affinity': '(Unsupervised NLP lexical affinity [0-100] to Mechanical Bearing, Gearbox, Shaft, Coupling & Roller failure tokens: BEARING, BRG, GEARBOX, COUPLING, ROLLER, PULLROD)',
    'nlp_electrical_motor_trip_affinity': '(Unsupervised NLP lexical affinity [0-100] to Electrical Motor, Overcurrent Trip, Contactor, LRS & VFD Drive tokens: MOTOR, TRIP, OVERCURRENT, LRS, CONTACTOR, VFD, CABLE)',
    'nlp_conveyor_belt_wear_affinity': '(Unsupervised NLP lexical affinity [0-100] to Conveyor Belt, Bucket Elevator, Chain, Chute Liner & Bag Filter wear tokens: BELT, BC, CHAIN, SPLICE, CHUTE, BAG, FILTER, DBC)',
    'nlp_hydraulic_pneumatic_leak_affinity': '(Unsupervised NLP lexical affinity [0-100] to Hydraulic, Pneumatic, Lube Oil, Seal & Valve leakage tokens: OIL, LEAKAGE, SEAL, POWER PACK, PISTON, BELLOW, SOV, VALVE, PUMP)',
    'nlp_instrumentation_plc_calib_affinity': '(Unsupervised NLP lexical affinity [0-100] to Instrumentation, Weighfeeder Calibration, Sensor & PLC fault tokens: CALIBRATION, PFISTER, PACKER, PROXY, SWITCH, RTD, PLC)',
    'maint_text_char_length': '(Total character length of SAP Plant Maintenance Order Short Text AUFK-KTEXT measuring breakdown description completeness)',
    'maint_text_word_count': '(Count of distinct alphanumeric word tokens in SAP Maintenance Order Text AUFK-KTEXT)',
    'conf_text_char_length': '(Total character length of SAP Maintenance Order Confirmation Text AFRU-LTXA1 capturing technician root-cause remarks)',
    'maint_text_digit_ratio': '(Proportion of numeric digits in AUFK-KTEXT capturing equipment tag numbers, kiln/mill unit IDs, and dimensions)',
    'maint_text_alpha_ratio': '(Proportion of alphabetic characters in AUFK-KTEXT capturing natural language failure mode description density)',
    'equipment_tag_count': '(Count of structured plant equipment tag codes extracted from AUFK-KTEXT such as 461HY110, 261BC110, 471RF100, 432KB101)',
    'failure_symptom_token_count': '(Count of explicit breakdown failure symptom tokens in maintenance text: BROKEN, DAMAGED, LEAKAGE, TRIPPED, WORN OUT, HIGH TEMP, OVERCURRENT)',
    'tfidf_active_ngrams': '(Number of non-zero TF-IDF unigram and bigram features activated by the combined maintenance text corpus)',
    'tfidf_semantic_energy': '(L1 norm of sublinear TF-IDF n-gram activations measuring semantic diagnostic density of the maintenance record)',
    'svd_semantic_comp_1': '(1st Latent Semantic Analysis TruncatedSVD component of the TF-IDF maintenance & confirmation text matrix)',
    'svd_semantic_comp_2': '(2nd Latent Semantic Analysis TruncatedSVD component separating mechanical/conveyor wear from electrical/instrumentation faults)',
    'svd_semantic_comp_3': '(3rd Latent Semantic Analysis TruncatedSVD component capturing kiln/raw-mill/cement-mill equipment subsystem semantics)',
    'svd_semantic_comp_4': '(4th Latent Semantic Analysis TruncatedSVD component capturing hydraulic/lube leakage and pneumatic valve maintenance patterns)',
    'svd_semantic_comp_5': '(5th Latent Semantic Analysis TruncatedSVD component capturing weighfeeder calibration and sensor replacement semantics)',
    'repeat_text_order_count': '(Number of SAP PM orders sharing the exact same normalized maintenance text indicating chronic recurring failures)',
    'equipment_order_frequency': '(Total historical maintenance order frequency on the extracted plant equipment tag or work center)',
    'confirmation_count': '(Total count of SAP PM operation confirmations logged in AFRU for the maintenance order)',
    'log_actual_duration_hrs': '(Log-stabilized actual maintenance execution and labor duration ln(1 + AFRU-IDAUR + ISM01 + ISM02))',
    'distinct_spares_count': '(Count of distinct spare part material numbers MATNR issued to the maintenance order in AUFM)',
    'log_spare_cost_inr': '(Log-stabilized total spare parts and component withdrawal cost in INR ln(1 + AUFM-DMBTR + RESB-ENWRT))',
    'order_turnaround_days': '(Maintenance order lifecycle turnaround time in days from Order Creation AUFK-ERDAT to Technical Completion IDAT2)',
    'is_breakdown_or_emergency_order': '(Binary indicator = 1 when Order Type is PM01 Breakdown Maintenance, Priority 1/2, or text contains acute failure symptoms)'
}


def compute_feature_importance(champion_model, feature_names: list, X_scaled: np.ndarray, y_true: np.ndarray = None):
    """
    Computes natural tree-based feature importance for Breakdown Root Cause Analysis across all 32 features.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if len(X_scaled) > 50000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(X_scaled), size=50000, replace=False)
        X_fit = X_scaled[sample_idx]
        y_fit = y_true[sample_idx] if y_true is not None else None
    else:
        X_fit = X_scaled
        y_fit = y_true

    if y_fit is not None and len(np.unique(y_fit)) > 1:
        model = RandomForestClassifier(n_estimators=100, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        model.fit(X_fit, y_fit)
        raw_importances = model.feature_importances_
    else:
        cluster_labels = champion_model.predict(X_fit[:, :2])
        model = RandomForestRegressor(n_estimators=100, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        model.fit(X_fit, cluster_labels)
        raw_importances = model.feature_importances_

    sort_indices = np.argsort(-raw_importances)
    sorted_features = [feature_names[i] for i in sort_indices]
    sorted_imps = raw_importances[sort_indices]

    explanations = [
        FEATURE_EXPLANATIONS.get(feat, f"(NLP & Plant Maintenance root-cause feature for {feat})")
        for feat in sorted_features
    ]

    importance_df = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_imps,
        'Direct Mathematical Influence On Breakdown Root Cause Extraction & PM Governance:': explanations
    })

    return importance_df
