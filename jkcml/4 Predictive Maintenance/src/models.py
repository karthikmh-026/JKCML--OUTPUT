"""
Machine Learning Models Module - Use Case 4: Predictive Maintenance Classification
(Predict Equipment Failures Before Breakdown)
Defines and trains 5 Unsupervised Machine Learning Models strictly on input features X:
1. Isolation Forest (Tree-based ensemble isolation of equipment operational anomaly and breakdown risk patterns)
2. Local Outlier Factor (LOF - Density-based local neighborhood anomaly detection with novelty=True)
3. One-Class SVM (Kernel support vector boundary novelty detection for machine operational signatures)
4. Gaussian Mixture Model (GMM - Probabilistic multivariate distribution log-likelihood anomaly score)
5. K-Means Clustering (Centroid-based distance from healthy preventive baseline)

After evaluating and ranking all 5 unsupervised models, selects the #1 Champion Model
for production scoring via SingleModelPredictiveMaintenanceClassifier.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


EQUIPMENT_HEALTH_TIERS = {
    0: "Healthy / Normal Operation",
    1: "Moderate Wear / Watchlist",
    2: "High Breakdown Risk",
    3: "Critical Failure Imminent"
}


class PredictiveMaintenanceUnsupervisedSuite:
    """
    Orchestrates the 100% unsupervised training, score calibration, and prediction
    of 5 Unsupervised ML Models for predictive maintenance.
    No target labels (Y) are ever used during training.
    """

    def __init__(self, contamination: float = 0.25, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.thresholds = {}
        self.meta = {}

    def _normalize_scores(self, raw_scores: np.ndarray, clip_percentiles: tuple = (1.0, 99.0)) -> np.ndarray:
        """Min-max normalizes scores into [0, 1] with percentile clipping."""
        p_low, p_high = np.percentile(raw_scores, clip_percentiles)
        clipped = np.clip(raw_scores, p_low, p_high)
        denom = max(1e-7, p_high - p_low)
        return (clipped - p_low) / denom

    def train_isolation_forest(self, X_train_scaled: np.ndarray):
        """
        Model 1: Isolation Forest
        Isolates equipment operating anomalies and severe failure risk patterns
        via recursive random partitioning.
        """
        print("[INFO] Training Unsupervised Model 1/5: Isolation Forest...")
        clf = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            max_samples=4096,
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_train_scaled)
        self.models['isolation_forest'] = clf

        # Sample for threshold computation if large
        eval_size = min(30000, len(X_train_scaled))
        raw = -clf.decision_function(X_train_scaled[:eval_size])
        train_scores = self._normalize_scores(raw)
        thresh = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))
        self.thresholds['isolation_forest'] = thresh
        self.meta['isolation_forest'] = {'p_low': float(np.percentile(raw, 1.0)), 'p_high': float(np.percentile(raw, 99.0))}
        return clf

    def train_lof(self, X_train_scaled: np.ndarray):
        """
        Model 2: Local Outlier Factor (LOF with novelty=True)
        Identifies machines with abnormal operational density compared to peer equipment.
        """
        print("[INFO] Training Unsupervised Model 2/5: Local Outlier Factor (LOF)...")
        clf = LocalOutlierFactor(
            n_neighbors=25,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1
        )
        max_samples = min(15000, len(X_train_scaled))
        rng = np.random.RandomState(self.random_state)
        sample_idx = rng.choice(len(X_train_scaled), size=max_samples, replace=False)
        clf.fit(X_train_scaled[sample_idx])
        self.models['lof'] = clf

        eval_size = min(20000, len(X_train_scaled))
        raw = -clf.decision_function(X_train_scaled[:eval_size])
        train_scores = self._normalize_scores(raw)
        thresh = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))
        self.thresholds['lof'] = thresh
        self.meta['lof'] = {'p_low': float(np.percentile(raw, 1.0)), 'p_high': float(np.percentile(raw, 99.0))}
        return clf

    def train_one_class_svm(self, X_train_scaled: np.ndarray):
        """
        Model 3: One-Class SVM
        Maps equipment operational signatures into kernel Hilbert space to learn
        the boundary of healthy equipment.
        """
        print("[INFO] Training Unsupervised Model 3/5: One-Class SVM...")
        clf = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=min(0.40, self.contamination)
        )
        max_samples = min(5000, len(X_train_scaled))
        rng = np.random.RandomState(self.random_state)
        sample_idx = rng.choice(len(X_train_scaled), size=max_samples, replace=False)
        clf.fit(X_train_scaled[sample_idx])
        self.models['one_class_svm'] = clf

        eval_size = min(15000, len(X_train_scaled))
        raw = -clf.decision_function(X_train_scaled[:eval_size])
        train_scores = self._normalize_scores(raw)
        thresh = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))
        self.thresholds['one_class_svm'] = thresh
        self.meta['one_class_svm'] = {'p_low': float(np.percentile(raw, 1.0)), 'p_high': float(np.percentile(raw, 99.0))}
        return clf

    def train_gmm(self, X_train_scaled: np.ndarray):
        """
        Model 4: Gaussian Mixture Model (GMM)
        Fits multivariate Gaussian distributions to equipment operating states;
        low log-likelihood signals operational degradation and breakdown risk.
        """
        print("[INFO] Training Unsupervised Model 4/5: Gaussian Mixture Model (GMM)...")
        sub_dim = min(8, X_train_scaled.shape[1])
        gmm = GaussianMixture(
            n_components=4,
            covariance_type='diag',
            n_init=3,
            max_iter=150,
            random_state=self.random_state
        )
        gmm.fit(X_train_scaled[:, :sub_dim])
        self.models['gmm'] = gmm

        raw = -gmm.score_samples(X_train_scaled[:, :sub_dim])
        train_scores = self._normalize_scores(raw)
        thresh = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))
        self.thresholds['gmm'] = thresh
        self.meta['gmm'] = {'p_low': float(np.percentile(raw, 1.0)), 'p_high': float(np.percentile(raw, 99.0)), 'sub_dim': sub_dim}
        return gmm

    def train_kmeans(self, X_train_scaled: np.ndarray):
        """
        Model 5: K-Means Clustering
        Partitions equipment operating profiles into 4 clusters. Identifies the most
        stable/healthy centroid and scores equipment breakdown risk by Euclidean distance.
        """
        print("[INFO] Training Unsupervised Model 5/5: K-Means Clustering...")
        km = KMeans(
            n_clusters=4,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=self.random_state
        )
        labels = km.fit_predict(X_train_scaled)
        self.models['kmeans'] = km

        # Find healthy baseline cluster centroid (the one closest to origin/median in RobustScaler space)
        centroid_dists = np.linalg.norm(km.cluster_centers_, axis=1)
        healthy_centroid_idx = int(np.argmin(centroid_dists))
        healthy_centroid = km.cluster_centers_[healthy_centroid_idx]

        raw = np.linalg.norm(X_train_scaled - healthy_centroid, axis=1)
        train_scores = self._normalize_scores(raw)
        thresh = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))
        self.thresholds['kmeans'] = thresh
        self.meta['kmeans'] = {
            'healthy_centroid': healthy_centroid,
            'p_low': float(np.percentile(raw, 1.0)),
            'p_high': float(np.percentile(raw, 99.0))
        }
        return km

    def train_all(self, X_train_scaled: np.ndarray):
        """Trains all 5 unsupervised models."""
        self.train_isolation_forest(X_train_scaled)
        self.train_lof(X_train_scaled)
        self.train_one_class_svm(X_train_scaled)
        self.train_gmm(X_train_scaled)
        self.train_kmeans(X_train_scaled)

    def predict_single(self, key: str, X_scaled: np.ndarray) -> dict:
        """Computes failure risk scores and binary predictions for a single model."""
        clf = self.models[key]
        meta = self.meta[key]
        thresh = self.thresholds[key]

        if key in ['isolation_forest', 'lof', 'one_class_svm']:
            raw = -clf.decision_function(X_scaled)
        elif key == 'gmm':
            sub_dim = meta.get('sub_dim', 8)
            raw = -clf.score_samples(X_scaled[:, :sub_dim])
        elif key == 'kmeans':
            healthy_c = meta['healthy_centroid']
            raw = np.linalg.norm(X_scaled - healthy_c, axis=1)
        else:
            raise ValueError(f"Unknown model key: {key}")

        p_low = meta['p_low']
        p_high = meta['p_high']
        clipped = np.clip(raw, p_low, p_high)
        denom = max(1e-7, p_high - p_low)
        scores = (clipped - p_low) / denom
        preds = (scores >= thresh).astype(int)

        return {'scores': scores, 'preds': preds, 'threshold': thresh}

    def predict_all(self, X_scaled: np.ndarray) -> dict:
        """Returns scores and predictions dictionary for all 5 models."""
        results = {}
        for key in self.models.keys():
            results[key] = self.predict_single(key, X_scaled)
        return results


class SingleModelPredictiveMaintenanceClassifier:
    """
    Production-ready standalone classifier packaging the #1 Champion Unsupervised Model,
    scaler, calibrated threshold, and health tier categorizer.
    """

    def __init__(self, model_key: str, model_instance, scaler, threshold: float, meta: dict, feature_cols: list):
        self.model_key = model_key
        self.model = model_instance
        self.scaler = scaler
        self.threshold = threshold
        self.meta = meta
        self.feature_cols = feature_cols

    def predict_risk_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes calibrated failure risk probability in [0, 1]."""
        if self.model_key in ['isolation_forest', 'lof', 'one_class_svm']:
            raw = -self.model.decision_function(X)
        elif self.model_key == 'gmm':
            sub_dim = self.meta.get('sub_dim', 8)
            raw = -self.model.score_samples(X[:, :sub_dim])
        elif self.model_key == 'kmeans':
            healthy_c = self.meta['healthy_centroid']
            raw = np.linalg.norm(X - healthy_c, axis=1)
        else:
            raise ValueError(f"Unknown model: {self.model_key}")

        p_low = self.meta['p_low']
        p_high = self.meta['p_high']
        clipped = np.clip(raw, p_low, p_high)
        denom = max(1e-7, p_high - p_low)
        return (clipped - p_low) / denom

    def predict_exception(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary equipment failure exception flag (1 = Failure Exception, 0 = Normal)."""
        scores = self.predict_risk_scores(X)
        return (scores >= self.threshold).astype(int)

    def assign_health_tier(self, score: float) -> str:
        """Assigns 4-tier operational health status."""
        if score >= 0.70:
            return "Critical Failure Imminent"
        elif score >= 0.45:
            return "High Breakdown Risk"
        elif score >= 0.25:
            return "Moderate Wear / Watchlist"
        else:
            return "Healthy / Normal Operation"

    def prescribe_action(self, row: dict) -> str:
        """Generates domain prescriptive maintenance recommendation based on operational indicators."""
        risk_score = row.get('failure_risk_score', 0.0)
        vibration = row.get('nlp_vibration_bearing_flag', 0.0)
        thermal = row.get('nlp_overheating_leakage_flag', 0.0)
        electrical = row.get('nlp_electrical_motor_flag', 0.0)
        wear = row.get('nlp_wear_damage_flag', 0.0)
        parts_cost = row.get('total_spare_parts_cost', 0.0)
        dur_var = row.get('duration_variance_days', 0.0)

        if risk_score >= 0.70:
            reasons = []
            if vibration > 0:
                reasons.append("Severe Bearing/Vibration Imbalance")
            if thermal > 0:
                reasons.append("Overheating/Oil Leakage Risk")
            if electrical > 0:
                reasons.append("Motor Tripping/Winding Fault")
            if wear > 0:
                reasons.append("Critical Structural Wear")
            if parts_cost > 50000:
                reasons.append("High Spare Replacement Cost")
            tag = " & ".join(reasons) if reasons else "Anomalous Operational Stress"
            return f"CRITICAL: Emergency Shutdown & Overhaul Recommended ({tag}). Halt operation to prevent catastrophic failure."
        elif risk_score >= 0.45:
            if vibration > 0 or thermal > 0:
                return "HIGH RISK: Schedule Immediate Condition Monitoring (Vibration Analysis & Thermography) within 48 hours."
            elif electrical > 0:
                return "HIGH RISK: Inspect Motor Starters, Relays, and Insulation Resistance before next production shift."
            elif dur_var > 7:
                return "HIGH RISK: Root Cause Investigation Required - Maintenance execution duration significantly exceeds plan."
            else:
                return "HIGH RISK: Priority Corrective Maintenance Required. Inspect key mechanical assemblies for premature wear."
        elif risk_score >= 0.25:
            return "WATCHLIST: Increase Lubrication & Inspection Frequency; monitor sensor telemetry during daily rounds."
        else:
            return "NORMAL: Maintain standard preventive maintenance schedule and routine operational checks."
