"""
Machine Learning Models Module - Use Case 21: Material Master Data Quality Scoring
Defines and trains the 5 Unsupervised Anomaly Detection models:
1. Isolation Forest
2. Local Outlier Factor (LOF with novelty=True)
3. One-Class SVM
4. DBSCAN (Density-Based Spatial Clustering)
5. Autoencoder (Bottleneck Neural Network Reconstruction)

Optimized to scale seamlessly to hundreds of thousands of records across HANA_P2 and HANA_P3.
After ranking all 5 unsupervised models, selects the #1 Champion Model (Isolation Forest)
for production scoring via SingleModelMaterialQualityDetector.
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
        Detects incomplete or incorrect material master records by recursively isolating
        multi-view governance, valuation, and completeness outliers.
        """
        print("[INFO] Training Unsupervised Model 1/5: Isolation Forest...")
        clf = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            max_samples=256,
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_train_scaled)
        self.models['isolation_forest'] = clf
        return clf

    def train_lof(self, X_train_scaled: np.ndarray):
        """
        Model 2: Local Outlier Factor (LOF)
        Detects material records with anomalous local density compared to peer materials.
        """
        print("[INFO] Training Unsupervised Model 2/5: Local Outlier Factor (LOF)...")
        clf = LocalOutlierFactor(
            n_neighbors=25,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1
        )
        X_sub = self._get_subsample(X_train_scaled, max_samples=20000)
        clf.fit(X_sub)
        self.models['lof'] = clf
        return clf

    def train_one_class_svm(self, X_train_scaled: np.ndarray):
        """
        Model 3: One-Class SVM
        Learns a normal material master governance hypersphere boundary using an RBF kernel.
        """
        print("[INFO] Training Unsupervised Model 3/5: One-Class SVM...")
        clf = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=self.contamination
        )
        X_sub = self._get_subsample(X_train_scaled, max_samples=20000)
        clf.fit(X_sub)
        self.models['one_class_svm'] = clf
        return clf

    def train_dbscan(self, X_train_scaled: np.ndarray, eps: float = 1.8, min_samples: int = 15):
        """
        Model 4: DBSCAN
        Identifies dense compliant material master clusters and flags noise points.
        """
        print("[INFO] Training Unsupervised Model 4/5: DBSCAN...")
        X_sub = self._get_subsample(X_train_scaled, max_samples=20000)
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X_sub)

        core_samples = X_sub[db.core_sample_indices_] if len(db.core_sample_indices_) > 0 else X_sub
        nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1).fit(core_samples)

        train_dist, _ = nn.kneighbors(X_sub[:min(5000, len(X_sub))])
        adaptive_eps = float(np.percentile(train_dist[:, -1], 100 * (1 - self.contamination)))

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
        Detects incomplete or inconsistent material master records via bottleneck reconstruction error.
        """
        print("[INFO] Training Unsupervised Model 5/5: Deep Neural Autoencoder...")
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(32, 16, 8, 16, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=512,
            learning_rate_init=0.005,
            max_iter=30,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state
        )
        autoencoder.fit(X_train_scaled, X_train_scaled)

        # Calibrate threshold on sample for efficiency
        X_cal = self._get_subsample(X_train_scaled, max_samples=30000)
        recon = autoencoder.predict(X_cal)
        train_errors = np.mean(np.square(X_cal - recon), axis=1)
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

    def predict_lof(self, X_scaled: np.ndarray, batch_size: int = 50000):
        clf = self.models['lof']
        batches = [X_scaled[i:i+batch_size] for i in range(0, len(X_scaled), batch_size)]
        preds = np.concatenate([(clf.predict(b) == -1).astype(int) for b in batches])
        scores = np.concatenate([-clf.decision_function(b) for b in batches])
        return preds, scores

    def predict_one_class_svm(self, X_scaled: np.ndarray, batch_size: int = 50000):
        clf = self.models['one_class_svm']
        batches = [X_scaled[i:i+batch_size] for i in range(0, len(X_scaled), batch_size)]
        preds = np.concatenate([(clf.predict(b) == -1).astype(int) for b in batches])
        scores = np.concatenate([-clf.decision_function(b) for b in batches])
        return preds, scores

    def predict_dbscan(self, X_scaled: np.ndarray, batch_size: int = 50000):
        info = self.models['dbscan']
        nn = info['nn']
        eps = info['eps']
        batches = [X_scaled[i:i+batch_size] for i in range(0, len(X_scaled), batch_size)]
        scores_list = []
        for b in batches:
            dists, _ = nn.kneighbors(b)
            scores_list.append(dists[:, -1])
        scores = np.concatenate(scores_list)
        preds = (scores >= eps).astype(int)
        return preds, scores

    def predict_autoencoder(self, X_scaled: np.ndarray, batch_size: int = 50000):
        autoencoder = self.models['autoencoder']
        threshold = self.thresholds['autoencoder']
        batches = [X_scaled[i:i+batch_size] for i in range(0, len(X_scaled), batch_size)]
        scores_list = []
        for b in batches:
            rec = autoencoder.predict(b)
            scores_list.append(np.mean(np.square(b - rec), axis=1))
        scores = np.concatenate(scores_list)
        preds = (scores >= threshold).astype(int)
        return preds, scores

    def predict_all(self, X_scaled: np.ndarray):
        """
        Runs inference across all 5 unsupervised models and returns predictions and anomaly scores.
        """
        results = {}
        p_if, s_if = self.predict_isolation_forest(X_scaled)
        results['Isolation Forest'] = {'preds': p_if, 'scores': s_if}

        p_ae, s_ae = self.predict_autoencoder(X_scaled)
        results['Autoencoder'] = {'preds': p_ae, 'scores': s_ae}

        p_svm, s_svm = self.predict_one_class_svm(X_scaled)
        results['One-Class SVM'] = {'preds': p_svm, 'scores': s_svm}

        p_db, s_db = self.predict_dbscan(X_scaled)
        results['DBSCAN'] = {'preds': p_db, 'scores': s_db}

        p_lof, s_lof = self.predict_lof(X_scaled)
        results['Local Outlier Factor'] = {'preds': p_lof, 'scores': s_lof}

        return results


class SingleModelMaterialQualityDetector:
    """
    Dedicated #1 Ranked Champion ML Model for Material Master Data Quality Scoring (Isolation Forest).
    Detects incomplete or incorrect SAP material master records across Basic Data (MARA/MAKT),
    Plant/MRP/Purchasing Views (MARC), Accounting/Valuation Views (MBEW), and UoM/Info Records (MARM/EINA).
    Computes:
    - Binary data quality anomaly predictions (1 for incomplete/incorrect material record, 0 for compliant)
    - Calibrated material quality anomaly probabilities in [0.0, 1.0]
    - Executive risk triage categories: Critical Risk, High Risk, Medium Risk, Low Risk
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42, pretrained_model=None):
        self.contamination = contamination
        self.random_state = random_state
        if pretrained_model is not None:
            self.model = pretrained_model
        else:
            self.model = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                max_samples=256,
                random_state=self.random_state,
                n_jobs=-1
            )
        self.min_score_ = None
        self.max_score_ = None
        self.offset_ = None

    def fit(self, X_scaled: np.ndarray):
        print("[INFO] Fitting Selected #1 Champion Model: Isolation Forest for Material Master Data Quality Scoring...")
        self.model.fit(X_scaled)
        # Sample for score calibration if dataset is massive
        cal_sub = X_scaled if len(X_scaled) <= 50000 else X_scaled[np.random.choice(len(X_scaled), 50000, replace=False)]
        raw_scores = -self.model.score_samples(cal_sub)
        self.min_score_ = float(np.percentile(raw_scores, 1))
        self.max_score_ = float(np.percentile(raw_scores, 99.5))
        self.offset_ = float(-self.model.offset_)
        print(f"[INFO] Champion Model fitted. Score range: [{self.min_score_:.4f}, {self.max_score_:.4f}], Anomaly Threshold: {self.offset_:.4f}")
        return self

    def predict(self, X_scaled: np.ndarray, batch_size: int = 50000) -> np.ndarray:
        if len(X_scaled) <= batch_size:
            return (self.model.predict(X_scaled) == -1).astype(int)
        batches = [X_scaled[i:i+batch_size] for i in range(0, len(X_scaled), batch_size)]
        preds = np.concatenate([(self.model.predict(b) == -1).astype(int) for b in batches])
        return preds

    def predict_proba(self, X_scaled: np.ndarray, batch_size: int = 50000):
        if len(X_scaled) <= batch_size:
            raw_scores = -self.model.score_samples(X_scaled)
        else:
            batches = [X_scaled[i:i+batch_size] for i in range(0, len(X_scaled), batch_size)]
            raw_scores = np.concatenate([-self.model.score_samples(b) for b in batches])

        scale = max(0.01, (self.max_score_ - self.min_score_) / 6.0) if self.max_score_ is not None else 0.1
        z = (raw_scores - self.offset_) / scale
        probs = 1.0 / (1.0 + np.exp(-z))
        return probs, raw_scores
