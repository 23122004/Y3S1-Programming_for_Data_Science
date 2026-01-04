import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from . import config

class Clusterer:
    def __init__(self, n_clusters: int,
                 label_col: str = 'Cluster',
                 labels: List[str] = None,
                 random_state: int = config.RANDOM_STATE):
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.label_col = label_col
        self.labels = labels
        self.scaler = StandardScaler()
        self.kmeans = None
        self.features = None
        self.label_map = None

    def fit_predict(self, df: pd.DataFrame,
                    features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit và dự đoán clusters."""
        self.features = features
        X = df[features].copy()

        # Scale và fit KMeans
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=config.KMEANS_N_INIT
        )
        clusters = self.kmeans.fit_predict(X_scaled)

        df_result = df.copy()

        # Gán labels nếu có
        if self.labels:
            self._assign_labels()
            df_result[self.label_col] = [self.label_map[c] for c in clusters]
        else:
            df_result[self.label_col] = clusters

        # Tính summary
        summary_cols = features + ['Task_Success_Rate']
        cluster_summary = df_result.groupby(self.label_col)[summary_cols].mean().round(2)

        return df_result, cluster_summary

    def _assign_labels(self):
        centroids_sum = [c.sum() for c in self.kmeans.cluster_centers_]
        sorted_indices = np.argsort(centroids_sum)[::-1]
        self.label_map = {idx: self.labels[rank] for rank, idx in enumerate(sorted_indices)}

class BasePredictor:
    def __init__(self, is_scale: bool = False):
        self.model = None
        self.features = None
        self.metrics = {}
        self.is_scale = is_scale
        self.scaler = StandardScaler() if is_scale else None

    def _compute_metrics(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict:
        return {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }

    def _print_metrics(self):
        print("Model Evaluation:")
        print(f"  R-squared: {self.metrics['r2_score']:.4f}")
        print(f"  RMSE: {self.metrics['rmse']:.4f}")
        print(f"  MAE: {self.metrics['mae']:.4f}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.features = list(X_train.columns)
        if self.is_scale:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")

        if self.is_scale:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        y_pred = self.predict(X_test)
        self.metrics = self._compute_metrics(y_test, y_pred)
        self._print_metrics()
        return self.metrics

class RandomForestPredictor(BasePredictor):
    def __init__(self, n_estimators: int = config.RF_N_ESTIMATORS,
                 random_state: int = config.RANDOM_STATE,
                 is_scale: bool = False):
        super().__init__(is_scale=is_scale)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )

    def get_feature_importances(self) -> Dict[str, float]:
        if self.features is None:
            raise ValueError("Model chưa được huấn luyện.")
        return dict(zip(self.features, self.model.feature_importances_))

class LinearRegressionPredictor(BasePredictor):
    def __init__(self, is_scale: bool = True):
        super().__init__(is_scale=is_scale)
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()

    def get_coefficients(self) -> Dict[str, float]:
        if self.features is None:
            raise ValueError("Model chưa được huấn luyện.")
        return dict(zip(self.features, self.model.coef_))