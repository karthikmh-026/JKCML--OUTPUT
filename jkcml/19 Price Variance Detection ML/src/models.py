"""
Machine Learning Models Module
Defines and trains the 5 Anomaly Detection models:
1. Isolation Forest
2. Local Outlier Factor (LOF with novelty=True)
3. One-Class SVM
4. DBSCAN (Density-Based Spatial Clustering)
5. Autoencoder (Bottleneck Neural Network Reconstruction)
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPRegressor


class AnomalyDetectionSuite:
    """
    Orchestrates the training, scoring, and prediction of 5 Anomaly Detection models.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.thresholds = {}

    def train_isolation_forest(self, X_train_scaled: np.ndarray):
        """
        Model 1: Isolation Forest
        Detects unusual purchase-price transactions by recursively isolating samples.
        """
        print("[INFO] Training Model 1: Isolation Forest...")
        clf = IsolationForest(
            n_estimators=150,
            contamination=self.contamination,
            max_samples='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_train_scaled)
        self.models['isolation_forest'] = clf
        return clf

    def train_lof(self, X_train_scaled: np.ndarray):
        """
        Model 2: Local Outlier Factor (LOF)
        Detects prices that are unusual compared with similar transactions in local density.
        """
        print("[INFO] Training Model 2: Local Outlier Factor (LOF)...")
        clf = LocalOutlierFactor(
            n_neighbors=25,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1
        )
        clf.fit(X_train_scaled)
        self.models['lof'] = clf
        return clf

    def train_one_class_svm(self, X_train_scaled: np.ndarray):
        """
        Model 3: One-Class SVM
        Learns normal purchasing-price patterns and identifies deviations using RBF kernel.
        """
        print("[INFO] Training Model 3: One-Class SVM...")
        clf = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=self.contamination
        )
        clf.fit(X_train_scaled)
        self.models['one_class_svm'] = clf
        return clf

    def train_dbscan(self, X_train_scaled: np.ndarray, eps: float = 1.5, min_samples: int = 15):
        """
        Model 4: DBSCAN
        Identifies normal price clusters and transactions outside those clusters.
        """
        print("[INFO] Training Model 4: DBSCAN...")
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X_train_scaled)
        
        # Store nearest neighbor model for out-of-sample test inference
        core_samples = X_train_scaled[db.core_sample_indices_] if len(db.core_sample_indices_) > 0 else X_train_scaled
        nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1).fit(core_samples)

        self.models['dbscan'] = {
            'dbscan': db,
            'nn': nn,
            'eps': eps,
            'train_labels': labels
        }
        return self.models['dbscan']

    def train_autoencoder(self, X_train_scaled: np.ndarray):
        """
        Model 5: Deep Neural Autoencoder
        Detects anomalies using reconstruction error from a bottleneck architecture.
        """
        print("[INFO] Training Model 5: Deep Neural Autoencoder...")
        # Deep bottleneck architecture: input -> 32 -> 16 -> 8 -> 16 -> 32 -> output
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(32, 16, 8, 16, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=128,
            learning_rate_init=0.005,
            max_iter=150,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state
        )
        # Train to reconstruct normal procurement inputs
        autoencoder.fit(X_train_scaled, X_train_scaled)

        # Determine anomaly threshold at (100 - contamination * 100) percentile of train reconstruction error
        train_recon = autoencoder.predict(X_train_scaled)
        train_errors = np.mean(np.square(X_train_scaled - train_recon), axis=1)
        threshold = np.percentile(train_errors, 100 * (1 - self.contamination))
        self.thresholds['autoencoder'] = threshold
        self.models['autoencoder'] = autoencoder
        return autoencoder

    def predict_isolation_forest(self, X_scaled: np.ndarray):
        clf = self.models['isolation_forest']
        # clf.predict returns 1 for inliers, -1 for outliers
        preds = (clf.predict(X_scaled) == -1).astype(int)
        # Decision function: lower values mean more anomalous; negate for positive anomaly score
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
        # Distance to k-th nearest core cluster point
        distances, _ = nn.kneighbors(X_scaled)
        k_distances = distances[:, -1]
        preds = (k_distances > eps).astype(int)
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
        Run inference across all 5 models and return dictionary of predictions and scores.
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


class SingleModelPriceVarianceDetector:
    """
    Dedicated single ML model for Price Variance Anomaly Detection using Isolation Forest.
    Finds unusual purchase order prices relative to contract, material, and group reference prices.
    Computes:
    - Binary price variance anomaly predictions (1 for anomaly, 0 for normal)
    - Calibrated price variance risk probabilities in [0.0, 1.0]
    - Executive risk triage categories: Critical Risk, High Risk, Medium Risk, Low Risk
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=150,
            contamination=self.contamination,
            max_samples='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.min_score_ = None
        self.max_score_ = None
        self.offset_ = None

    def fit(self, X_scaled: np.ndarray):
        print("[INFO] Training Single ML Champion Model: Isolation Forest for Price Variance Detection...")
        self.model.fit(X_scaled)
        raw_scores = -self.model.score_samples(X_scaled)
        self.min_score_ = float(np.percentile(raw_scores, 1))
        self.max_score_ = float(np.percentile(raw_scores, 99.5))
        self.offset_ = float(-self.model.offset_)
        print(f"[INFO] Model fitted. Score range: [{self.min_score_:.4f}, {self.max_score_:.4f}], Anomaly Threshold: {self.offset_:.4f}")
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
