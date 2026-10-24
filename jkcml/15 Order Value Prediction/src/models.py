"""
Machine Learning Models Module - Use Case 15: Order Value Prediction
Category: Regression
Objective: Predict next order value of customers.

Defines and trains 5 Supervised Machine Learning Regression Models:
1. Random Forest Regressor (Ensemble of decision trees with feature bagging)
2. Gradient Boosting Regressor (Sequential gradient boosted residual trees)
3. HistGradientBoosting Regressor (Histogram-based gradient tree boosting)
4. Decision Tree Regressor (Non-linear recursive partitioning tree)
5. Ridge Regressor (L2 regularized linear model)

After evaluating and ranking all 5 regression models on the held-out test split,
selects the #1 Champion Model (Random Forest Regressor) for production scoring
via SingleModelOrderValueRegressor.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
import joblib


class OrderValueRegressionSuite:
    """
    Orchestrates the training, scoring, and prediction of 5 Supervised Regression Models
    for predicting the next order value of customers.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray):
        """Model 1: Random Forest Regressor"""
        print("[INFO] Training Regression Model 1/5: Random Forest Regressor...")
        model = RandomForestRegressor(
            n_estimators=60,
            max_depth=14,
            max_samples=0.25 if len(X_train) > 50000 else None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model

    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray):
        """Model 2: Gradient Boosting Regressor"""
        print("[INFO] Training Regression Model 2/5: Gradient Boosting Regressor...")
        model = GradientBoostingRegressor(
            n_estimators=80,
            max_depth=5,
            subsample=0.30 if len(X_train) > 50000 else 1.0,
            learning_rate=0.1,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        return model

    def train_hist_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray):
        """Model 3: HistGradientBoosting Regressor"""
        print("[INFO] Training Regression Model 3/5: HistGradientBoosting Regressor...")
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['hist_gradient_boosting'] = model
        return model

    def train_decision_tree(self, X_train: np.ndarray, y_train: np.ndarray):
        """Model 4: Decision Tree Regressor"""
        print("[INFO] Training Regression Model 4/5: Decision Tree Regressor...")
        model = DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['decision_tree'] = model
        return model

    def train_ridge(self, X_train: np.ndarray, y_train: np.ndarray):
        """Model 5: Ridge Regressor"""
        print("[INFO] Training Regression Model 5/5: Ridge Regressor...")
        model = Ridge(alpha=10.0, random_state=self.random_state)
        model.fit(X_train, y_train)
        self.models['ridge'] = model
        return model

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains all 5 regression models."""
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        self.train_hist_gradient_boosting(X_train, y_train)
        self.train_decision_tree(X_train, y_train)
        self.train_ridge(X_train, y_train)
        return self.models

    def predict_all(self, X_test: np.ndarray) -> dict:
        """
        Generates predictions across all 5 models (clipping negative values to zero).
        """
        results = {}
        for key, model in self.models.items():
            raw_preds = model.predict(X_test)
            results[key] = np.maximum(0.0, raw_preds)
        return results


class SingleModelOrderValueRegressor:
    """
    Production container wrapping the selected #1 Champion Regression Model.
    Predicts next order values of customers and provides explainability.
    """

    def __init__(self, random_state: int = 42, pretrained_model=None):
        self.random_state = random_state
        self.model = pretrained_model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits or refines the champion regression model."""
        if self.model is None:
            self.model = RandomForestRegressor(
                n_estimators=80,
                max_depth=14,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts continuous next order values (INR)."""
        preds = self.model.predict(X)
        return np.maximum(0.0, preds)

    def save(self, filepath: str):
        """Saves the fitted model artifact."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Loads a saved pipeline from disk."""
        return joblib.load(filepath)
