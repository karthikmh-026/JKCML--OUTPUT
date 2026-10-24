"""
Machine Learning Models Module - Use Case 8: Quality Inspection Failure Prediction
Defines and trains the 5 Unsupervised Anomaly/Failure Detection ML models:
1. Isolation Forest (Tree-partition outlier isolation)
2. Local Outlier Factor (LOF with novelty=True density estimation)
3. One-Class SVM (RBF hypersphere boundary learning)
4. DBSCAN (Density-Based Spatial Clustering with k-NN distance scoring)
5. Deep Neural Autoencoder (Bottleneck neural reconstruction error)

After ranking all 5 unsupervised models, deploys the #1 Ranked Champion Model
via SingleModelQualityFailureDetector for production-grade inference, risk triage,
and calibrated failure probability scoring across 100% of the dataset.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPRegressor


class QualityInspectionFailureDetectionSuite:
    """
    Orchestrates the unsupervised training, scoring, and prediction across 5 ML models.
    No target output labels (Y) are ever used during training (100% unsupervised).
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.thresholds = {}

    def _get_subsample(self, X: np.ndarray, max_rows: int = 25000) -> np.ndarray:
        """Returns a deterministic random subsample for O(N^2) models if dataset is large."""
        if len(X) <= max_rows:
            return X
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        return X[idx]

    def train_isolation_forest(self, X_train_scaled: np.ndarray):
        """
        Model 1: Isolation Forest
        Detects anomalous inspection lots by recursively partitioning features in random trees.
        Scales linearly and trains across all data points.
        """
        print("[INFO] Training Unsupervised Model 1/5: Isolation Forest...")
        clf = IsolationForest(
            n_estimators=120,
            contamination=self.contamination,
            max_samples=min(2048, len(X_train_scaled)),
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_train_scaled)
        self.models['isolation_forest'] = clf
        return clf

    def train_lof(self, X_train_scaled: np.ndarray):
        """
        Model 2: Local Outlier Factor (LOF)
        Measures the local density of an inspection lot relative to surrounding lots.
        """
        print("[INFO] Training Unsupervised Model 2/5: Local Outlier Factor (LOF)...")
        sub_X = self._get_subsample(X_train_scaled, max_rows=25000)
        clf = LocalOutlierFactor(
            n_neighbors=25,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1
        )
        clf.fit(sub_X)
        self.models['lof'] = clf
        return clf

    def train_one_class_svm(self, X_train_scaled: np.ndarray):
        """
        Model 3: One-Class SVM
        Learns an enclosing decision boundary around normal inspection lots in RBF kernel space.
        """
        print("[INFO] Training Unsupervised Model 3/5: One-Class SVM...")
        sub_X = self._get_subsample(X_train_scaled, max_rows=25000)
        clf = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=self.contamination
        )
        clf.fit(sub_X)
        self.models['one_class_svm'] = clf
        return clf

    def train_dbscan(self, X_train_scaled: np.ndarray, eps: float = 2.0, min_samples: int = 15):
        """
        Model 4: DBSCAN (Density-Based Spatial Clustering)
        Identifies dense clusters of normal inspection lots and flags outlier noise lots.
        """
        print("[INFO] Training Unsupervised Model 4/5: DBSCAN...")
        sub_X = self._get_subsample(X_train_scaled, max_rows=25000)
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(sub_X)

        core_samples = sub_X[db.core_sample_indices_] if len(db.core_sample_indices_) > 0 else sub_X
        nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1).fit(core_samples)

        distances, _ = nn.kneighbors(sub_X)
        k_distances = distances[:, -1]
        adaptive_eps = float(np.percentile(k_distances, 100 * (1 - self.contamination)))

        self.models['dbscan'] = {
            'dbscan': db,
            'nn': nn,
            'eps': adaptive_eps,
            'train_labels': labels
        }
        return self.models['dbscan']

    def train_autoencoder(self, X_train_scaled: np.ndarray):
        """
        Model 5: Deep Neural Autoencoder
        Encodes inspection features through a multi-layer bottleneck and reconstructs them.
        """
        print("[INFO] Training Unsupervised Model 5/5: Deep Neural Autoencoder...")
        sub_X = self._get_subsample(X_train_scaled, max_rows=40000)
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(32, 16, 8, 16, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=256,
            learning_rate_init=0.005,
            max_iter=60,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state
        )
        autoencoder.fit(sub_X, sub_X)

        train_recon = autoencoder.predict(sub_X)
        train_errors = np.mean(np.square(sub_X - train_recon), axis=1)
        threshold = float(np.percentile(train_errors, 100 * (1 - self.contamination)))
        self.thresholds['autoencoder'] = threshold
        self.models['autoencoder'] = autoencoder
        return autoencoder

    def train_all(self, X_train_scaled: np.ndarray):
        """Trains all 5 unsupervised models."""
        self.train_isolation_forest(X_train_scaled)
        self.train_lof(X_train_scaled)
        self.train_one_class_svm(X_train_scaled)
        self.train_dbscan(X_train_scaled)
        self.train_autoencoder(X_train_scaled)
        return self.models

    def predict_isolation_forest(self, X_scaled: np.ndarray):
        clf = self.models['isolation_forest']
        preds = (clf.predict(X_scaled) == -1).astype(int)
        scores = -clf.score_samples(X_scaled)
        return preds, scores

    def predict_lof(self, X_scaled: np.ndarray):
        clf = self.models['lof']
        preds = (clf.predict(X_scaled) == -1).astype(int)
        scores = -clf.decision_function(X_scaled)
        return preds, scores

    def predict_one_class_svm(self, X_scaled: np.ndarray):
        clf = self.models['one_class_svm']
        preds = (clf.predict(X_scaled) == -1).astype(int)
        scores = -clf.decision_function(X_scaled)
        return preds, scores

    def predict_dbscan(self, X_scaled: np.ndarray):
        info = self.models['dbscan']
        nn = info['nn']
        eps = info['eps']
        distances, _ = nn.kneighbors(X_scaled)
        k_distances = distances[:, -1]
        preds = (k_distances >= eps).astype(int)
        scores = k_distances
        return preds, scores

    def predict_autoencoder(self, X_scaled: np.ndarray):
        autoencoder = self.models['autoencoder']
        threshold = self.thresholds['autoencoder']
        reconstructed = autoencoder.predict(X_scaled)
        mse_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
        preds = (mse_errors >= threshold).astype(int)
        scores = mse_errors
        return preds, scores

    def predict_all(self, X_scaled: np.ndarray):
        """Runs inference across all 5 models and returns predictions and raw anomaly scores."""
        results = {}
        p_if, s_if = self.predict_isolation_forest(X_scaled)
        results['Isolation Forest'] = {'preds': p_if, 'scores': s_if}

        p_lof, s_lof = self.predict_lof(X_scaled)
        results['Local Outlier Factor'] = {'preds': p_lof, 'scores': s_lof}

        p_svm, s_svm = self.predict_one_class_svm(X_scaled)
        results['One-Class SVM'] = {'preds': p_svm, 'scores': s_svm}

        p_db, s_db = self.predict_dbscan(X_scaled)
        results['DBSCAN'] = {'preds': p_db, 'scores': s_db}

        p_ae, s_ae = self.predict_autoencoder(X_scaled)
        results['Autoencoder'] = {'preds': p_ae, 'scores': s_ae}

        return results


class SingleModelQualityFailureDetector:
    """
    Dedicated #1 Ranked Champion ML Model for Quality Inspection Failure Prediction.
    Computes:
    - Binary failure exception predictions (1 for failure/risk exception, 0 for normal pass)
    - Calibrated failure probabilities P(Failure) in [0.0, 1.0]
    - Executive risk triage categories: Critical Failure Risk, High Risk, Medium Risk, Low Risk
    """

    def __init__(self, model_name: str = "Isolation Forest", contamination: float = 0.05, random_state: int = 42, pretrained_model=None):
        self.model_name = model_name
        self.contamination = contamination
        self.random_state = random_state
        if pretrained_model is not None:
            self.model = pretrained_model
        else:
            if "Isolation" in model_name:
                self.model = IsolationForest(
                    n_estimators=150,
                    contamination=self.contamination,
                    max_samples=min(2048, 100000),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif "Local Outlier" in model_name or "LOF" in model_name:
                self.model = LocalOutlierFactor(
                    n_neighbors=25,
                    contamination=self.contamination,
                    novelty=True,
                    n_jobs=-1
                )
            elif "SVM" in model_name:
                self.model = OneClassSVM(
                    kernel='rbf',
                    gamma='scale',
                    nu=self.contamination
                )
            else:
                self.model = IsolationForest(
                    n_estimators=150,
                    contamination=self.contamination,
                    max_samples=min(2048, 100000),
                    random_state=self.random_state,
                    n_jobs=-1
                )
        self.min_score_ = None
        self.max_score_ = None
        self.offset_ = None

    def fit(self, X_scaled: np.ndarray):
        print(f"[INFO] Fitting Selected #1 Champion Model ({self.model_name}) for Production Scoring...")
        self.model.fit(X_scaled)
        if hasattr(self.model, 'score_samples'):
            # In IsolationForest, score_samples is negative; higher -score_samples means more anomalous
            raw_scores = -self.model.score_samples(X_scaled)
            self.offset_ = float(-self.model.offset_) if hasattr(self.model, 'offset_') else float(np.percentile(raw_scores, 100 * (1 - self.contamination)))
        elif hasattr(self.model, 'decision_function'):
            raw_scores = -self.model.decision_function(X_scaled)
            self.offset_ = 0.0
        else:
            raw_scores = np.zeros(len(X_scaled))
            self.offset_ = 0.0

        self.min_score_ = float(np.percentile(raw_scores, 1))
        self.max_score_ = float(np.percentile(raw_scores, 99.5))
        print(f"[INFO] Champion Model fitted. Score range: [{self.min_score_:.4f}, {self.max_score_:.4f}], Offset Threshold: {self.offset_:.4f}")
        return self

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        raw_preds = self.model.predict(X_scaled)
        return (raw_preds == -1).astype(int)

    def predict_proba(self, X_scaled: np.ndarray):
        if hasattr(self.model, 'score_samples'):
            raw_scores = -self.model.score_samples(X_scaled)
        elif hasattr(self.model, 'decision_function'):
            raw_scores = -self.model.decision_function(X_scaled)
        else:
            raw_scores = np.zeros(len(X_scaled))

        scale = max(0.005, (self.max_score_ - self.min_score_) / 6.0) if self.max_score_ is not None else 0.1
        z = (raw_scores - self.offset_) / scale
        probs = 1.0 / (1.0 + np.exp(-z))
        return probs, raw_scores

    def assign_risk_tiers(self, probs: np.ndarray) -> np.ndarray:
        tiers = np.empty(len(probs), dtype=object)
        for i, p in enumerate(probs):
            if p >= 0.70:
                tiers[i] = "Critical Failure Risk"
            elif p >= 0.50:
                tiers[i] = "High Risk"
            elif p >= 0.30:
                tiers[i] = "Medium Risk"
            else:
                tiers[i] = "Low Risk"
        return tiers
