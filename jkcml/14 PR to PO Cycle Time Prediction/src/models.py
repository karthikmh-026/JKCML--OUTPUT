"""
Machine Learning Models Module - Use Case 14: PR to PO Cycle Time Prediction
Category: Regression
Objective: Estimate procurement processing time.

Defines and trains the 5 Regression ML models:
1. Hist Gradient Boosting Regressor (LightGBM-style histogram booster optimized for large datasets)
2. Extra Trees Regressor
3. Random Forest Regressor
4. Bayesian Ridge Regressor
5. Ridge Regression

After ranking all 5 regression models on R² Score, MAE, RMSE, and Accuracy %,
selects the #1 Champion Model for production inference via SingleModelPRCycleRegressor.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, BayesianRidge
import joblib


class PRCycleRegressionSuite:
    """
    Orchestrates the training, scoring, and evaluation of 5 Regression ML models
    to predict PR to PO Cycle Time (procurement processing time in days).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}

    def train_hist_gradient_boosting(self, X_train_scaled: np.ndarray, y_train: np.ndarray):
        """
        Model 1: Hist Gradient Boosting Regressor
        Histogram-based gradient boosted trees engineered specifically for massive tabular ERP datasets.
        """
        print("[INFO] Training Regression Model 1/5: Hist Gradient Boosting Regressor...")
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=self.random_state
        )
        model.fit(X_train_scaled, y_train)
        self.models['Hist Gradient Boosting'] = model
        return model

    def train_extra_trees(self, X_train_scaled: np.ndarray, y_train: np.ndarray):
        """
        Model 2: Extra Trees Regressor
        Extremely randomized trees reducing variance across high-dimensional procurement features.
        """
        print("[INFO] Training Regression Model 2/5: Extra Trees Regressor...")
        model = ExtraTreesRegressor(
            n_estimators=30,
            max_depth=12,
            max_samples=0.25,
            bootstrap=True,
            min_samples_split=5,
            min_samples_leaf=4,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        self.models['Extra Trees'] = model
        return model

    def train_random_forest(self, X_train_scaled: np.ndarray, y_train: np.ndarray):
        """
        Model 3: Random Forest Regressor
        Ensemble of bagged decision trees modeling non-linear lead time interactions.
        """
        print("[INFO] Training Regression Model 3/5: Random Forest Regressor...")
        model = RandomForestRegressor(
            n_estimators=30,
            max_depth=12,
            max_samples=0.25,
            bootstrap=True,
            min_samples_split=5,
            min_samples_leaf=4,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        self.models['Random Forest'] = model
        return model

    def train_bayesian_ridge(self, X_train_scaled: np.ndarray, y_train: np.ndarray):
        """
        Model 4: Bayesian Ridge Regressor
        Probabilistic linear regression estimating regularized posterior cycle time distributions.
        """
        print("[INFO] Training Regression Model 4/5: Bayesian Ridge Regressor...")
        model = BayesianRidge()
        model.fit(X_train_scaled, y_train)
        self.models['Bayesian Ridge'] = model
        return model

    def train_ridge(self, X_train_scaled: np.ndarray, y_train: np.ndarray):
        """
        Model 5: Ridge Regression
        L2-penalized linear model establishing continuous baseline coefficient relationships.
        """
        print("[INFO] Training Regression Model 5/5: Ridge Regression...")
        model = Ridge(alpha=1.0, random_state=self.random_state)
        model.fit(X_train_scaled, y_train)
        self.models['Ridge Regression'] = model
        return model

    def train_all(self, X_train_scaled: np.ndarray, y_train: np.ndarray):
        """Trains all 5 regression models."""
        self.train_hist_gradient_boosting(X_train_scaled, y_train)
        self.train_extra_trees(X_train_scaled, y_train)
        self.train_random_forest(X_train_scaled, y_train)
        self.train_bayesian_ridge(X_train_scaled, y_train)
        self.train_ridge(X_train_scaled, y_train)
        return self.models

    def predict_all(self, X_test_scaled: np.ndarray) -> dict:
        """
        Generates procurement processing time predictions across all 5 models.
        Predictions are clipped to >= 0.0 (cycle time cannot be negative).
        """
        results = {}
        for name, model in self.models.items():
            preds = model.predict(X_test_scaled)
            results[name] = np.clip(preds, 0.0, None)
        return results


class SingleModelPRCycleRegressor:
    """
    Dedicated production inference class for the #1 Champion Regression Model.
    Provides estimated procurement processing time predictions in days.
    """

    def __init__(self, champion_name: str = "Hist Gradient Boosting", pretrained_model=None):
        self.champion_name = champion_name
        self.model = pretrained_model

    def fit(self, X_train_scaled: np.ndarray, y_train: np.ndarray):
        """Fits the champion regression model."""
        if self.model is None:
            self.model = HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
        self.model.fit(X_train_scaled, y_train)
        return self

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predicts estimated procurement processing time in days (clipped >= 0.0)."""
        raw_preds = self.model.predict(X_scaled)
        return np.clip(raw_preds, 0.0, None)

    def save(self, filepath: str):
        """Serializes the regressor to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Loads serialized regressor from disk."""
        return joblib.load(filepath)
