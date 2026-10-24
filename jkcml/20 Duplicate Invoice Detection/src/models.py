"""
Machine Learning Models Module - Use Case 20: Duplicate Invoice Detection
Defines and trains the 5 Unsupervised Anomaly Detection models:
1. Isolation Forest
2. Local Outlier Factor (LOF with novelty=True)
3. One-Class SVM
4. DBSCAN (Density-Based Spatial Clustering)
5. Autoencoder (Bottleneck Neural Network Reconstruction)

After ranking all 5 unsupervised models, selects the #1 Champion Model (Isolation Forest)
for production scoring via SingleModelDuplicateInvoiceDetector.
Optimized for high-volume enterprise data (100k+ to 1M+ transactions).
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPRegressor


class AnomalyDetectionSuite:
    """
    Orchestrates the unsupervised training, scoring, and prediction of 5 Anomaly Detection models.
    No target output labels (Y) are ever used during training (100% unsupervised).
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.thresholds = {}

    def _get_subsample(self, X: np.ndarray, max_samples: int = 20000) -> np.ndarray:
        if len(X) <= max_samples:
            return X
        np.random.seed(self.random_state)
        idx = np.random.choice(len(X), size=max_samples, replace=False)
        return X[idx]

    def train_isolation_forest(self, X_train_scaled: np.ndarray):
        """
        Model 1: Isolation Forest
        Detects duplicate invoice anomalies by recursively isolating multi-key collision outliers.
        """
        print("[INFO] Training Unsupervised Model 1/5: Isolation Forest...")
        max_samples = min(2048, len(X_train_scaled))
        clf = IsolationForest(
            n_estimators=120,
            contamination=self.contamination,
            max_samples=max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_train_scaled)
        self.models['isolation_forest'] = clf
        return clf

    def train_lof(self, X_train_scaled: np.ndarray):
        """
        Model 2: Local Outlier Factor (LOF)
        Detects invoices with anomalous local density compared to neighboring vendor transactions.
        """
        print("[INFO] Training Unsupervised Model 2/5: Local Outlier Factor (LOF)...")
        X_fit = self._get_subsample(X_train_scaled, max_samples=20000)
        clf = LocalOutlierFactor(
            n_neighbors=25,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1
        )
        clf.fit(X_fit)
        self.models['lof'] = clf
        return clf

    def train_one_class_svm(self, X_train_scaled: np.ndarray):
        """
        Model 3: One-Class SVM
        Learns a normal invoice hypersphere boundary using an RBF kernel and flags exterior points.
        """
        print("[INFO] Training Unsupervised Model 3/5: One-Class SVM...")
        X_fit = self._get_subsample(X_train_scaled, max_samples=15000)
        clf = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=self.contamination
        )
        clf.fit(X_fit)
        self.models['one_class_svm'] = clf
        return clf

    def train_dbscan(self, X_train_scaled: np.ndarray, eps: float = 1.8, min_samples: int = 15):
        """
        Model 4: DBSCAN
        Identifies dense normal invoice clusters and flags noise points outside core neighborhoods.
        """
        print("[INFO] Training Unsupervised Model 4/5: DBSCAN...")
        X_fit = self._get_subsample(X_train_scaled, max_samples=20000)
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X_fit)

        core_samples = X_fit[db.core_sample_indices_] if len(db.core_sample_indices_) > 0 else X_fit
        nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1).fit(core_samples)

        distances, _ = nn.kneighbors(X_fit)
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
        Detects duplicate invoice anomalies via bottleneck neural network reconstruction error.
        """
        print("[INFO] Training Unsupervised Model 5/5: Deep Neural Autoencoder...")
        X_fit = self._get_subsample(X_train_scaled, max_samples=30000)
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
        autoencoder.fit(X_fit, X_fit)

        train_recon = autoencoder.predict(X_fit)
        train_errors = np.mean(np.square(X_fit - train_recon), axis=1)
        threshold = float(np.percentile(train_errors, 100 * (1 - self.contamination)))
        self.thresholds['autoencoder'] = threshold
        self.models['autoencoder'] = autoencoder
        return autoencoder

    def train_all(self, X_train_scaled: np.ndarray):
        """Trains all 5 unsupervised anomaly detection models."""
        self.train_isolation_forest(X_train_scaled)
        self.train_lof(X_train_scaled)
        self.train_one_class_svm(X_train_scaled)
        self.train_dbscan(X_train_scaled)
        self.train_autoencoder(X_train_scaled)
        return self.models

    def predict_isolation_forest(self, X_scaled: np.ndarray):
        clf = self.models['isolation_forest']
        preds = (clf.predict(X_scaled) == -1).astype(int)
        scores = -clf.decision_function(X_scaled)
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
        """
        Runs inference across all 5 unsupervised models and returns predictions and anomaly scores.
        """
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


class SingleModelDuplicateInvoiceDetector:
    """
    Dedicated #1 Ranked Champion ML Model for Duplicate Invoice Detection (Isolation Forest).
    Identifies potential duplicate payments across vendor reference numbers, amounts, dates, and PO items.
    Computes:
    - Binary duplicate anomaly predictions (1 for duplicate risk, 0 for normal)
    - Calibrated duplicate invoice probabilities in [0.0, 1.0]
    - Executive risk triage categories: Critical Risk, High Risk, Medium Risk, Low Risk
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42, pretrained_model=None):
        self.contamination = contamination
        self.random_state = random_state
        if pretrained_model is not None:
            self.model = pretrained_model
        else:
            self.model = IsolationForest(
                n_estimators=120,
                contamination=self.contamination,
                max_samples=2048,
                random_state=self.random_state,
                n_jobs=-1
            )
        self.min_score_ = None
        self.max_score_ = None
        self.offset_ = None

    def fit(self, X_scaled: np.ndarray):
        print("[INFO] Fitting Selected #1 Champion Model: Isolation Forest for Duplicate Invoice Detection...")
        self.model.fit(X_scaled)
        raw_scores = -self.model.score_samples(X_scaled)
        self.min_score_ = float(np.percentile(raw_scores, 1))
        self.max_score_ = float(np.percentile(raw_scores, 99.5))
        self.offset_ = float(-self.model.offset_)
        print(f"[INFO] Champion Model fitted. Score range: [{self.min_score_:.4f}, {self.max_score_:.4f}], Anomaly Threshold: {self.offset_:.4f}")
        return self

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        raw_preds = self.model.predict(X_scaled)
        return (raw_preds == -1).astype(int)

    def predict_proba(self, X_scaled: np.ndarray):
        raw_scores = -self.model.score_samples(X_scaled)
        scale = max(0.01, (self.max_score_ - self.min_score_) / 6.0) if self.max_score_ is not None else 0.1
        z = (raw_scores - self.offset_) / scale
        probs = 1.0 / (1.0 + np.exp(-z))
        return probs, raw_scores
