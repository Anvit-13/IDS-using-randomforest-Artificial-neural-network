"""
MLPipeline Module for Network Anomaly Detection and GAN Dataset Preparation.

Handles feature selection, normalization with StandardScaler, unsupervised machine learning
(Isolation Forest, One-Class SVM, DBSCAN, KMeans), anomaly score evaluation, and model persistence.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.logger import setup_logger

logger = setup_logger("MLPipeline")


class MLPipeline:
    """Class responsible for ML feature preprocessing, model training, evaluation, and persistence."""

    def __init__(
        self,
        flow_df: pd.DataFrame,
        models_dir: Path = Path("models"),
        config_params: Dict = None,
    ) -> None:
        """Initializes MLPipeline.

        Args:
            flow_df: Extracted flow DataFrame.
            models_dir: Directory to save serialized models.
            config_params: Hyperparameters for ML models from configuration.
        """
        self.flow_df = flow_df.copy()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.config_params = config_params or {}

        self.scaler: StandardScaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.X_scaled: np.ndarray = np.array([])
        self.anomaly_results: pd.DataFrame = pd.DataFrame()
        self.trained_models: Dict[str, Any] = {}

    def _select_and_preprocess_features(self) -> np.ndarray:
        """Selects numerical columns, handles missing/infinite values, and standardizes features."""
        # Select numeric columns, excluding port numbers and raw metadata identifiers
        exclude_cols = {"FlowID", "Source_IP", "Destination_IP", "Source_Port", "Destination_Port", "Protocol"}
        numeric_cols = self.flow_df.select_dtypes(include=[np.number]).columns
        self.feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        logger.info(f"Selected {len(self.feature_cols)} numerical flow features for ML processing.")

        X = self.flow_df[self.feature_cols].copy()

        # Handle NaNs and Infs
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)
        X.fillna(0.0, inplace=True)

        # Scale features
        self.X_scaled = self.scaler.fit_transform(X)

        # Save scaler for future GAN dataset normalization inference
        scaler_path = self.models_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved StandardScaler model to '{scaler_path}'.")

        return self.X_scaled

    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs the entire feature scaling, model training, and anomaly prediction pipeline.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - GAN-ready scaled flow features DataFrame.
                - Anomaly evaluation report DataFrame.
        """
        logger.info("Executing ML Anomaly Detection Pipeline...")

        self._select_and_preprocess_features()

        # Construct GAN-Ready clean DataFrame
        gan_ready_df = pd.DataFrame(self.X_scaled, columns=self.feature_cols)
        gan_ready_df.insert(0, "FlowID", self.flow_df["FlowID"].values)

        # Initialize Anomaly Results Container
        results_df = pd.DataFrame({"FlowID": self.flow_df["FlowID"].values})

        # Train Models
        results_df["IsoForest_Pred"] = self._train_isolation_forest()
        results_df["OneClassSVM_Pred"] = self._train_one_class_svm()
        results_df["DBSCAN_Cluster"] = self._train_dbscan()
        results_df["DBSCAN_Anomaly"] = np.where(results_df["DBSCAN_Cluster"] == -1, -1, 1)
        results_df["KMeans_Cluster"] = self._train_kmeans()

        # Consensus Anomaly Score (-1 indicates anomaly by majority vote)
        anomaly_votes = (
            (results_df["IsoForest_Pred"] == -1).astype(int) +
            (results_df["OneClassSVM_Pred"] == -1).astype(int) +
            (results_df["DBSCAN_Anomaly"] == -1).astype(int)
        )
        results_df["Consensus_Anomaly"] = np.where(anomaly_votes >= 2, -1, 1)

        self.anomaly_results = results_df
        logger.info("ML Anomaly Detection Pipeline Completed Successfully.")
        return gan_ready_df, self.anomaly_results

    def _train_isolation_forest(self) -> np.ndarray:
        """Trains Isolation Forest model."""
        params = self.config_params.get("isolation_forest", {"n_estimators": 100, "contamination": 0.05, "random_state": 42})
        logger.info(f"Training Isolation Forest model (params: {params})...")

        model = IsolationForest(**params)
        preds = model.fit_predict(self.X_scaled)

        model_path = self.models_dir / "isolation_forest.joblib"
        joblib.dump(model, model_path)
        self.trained_models["isolation_forest"] = model
        logger.info(f"Saved Isolation Forest to '{model_path}'. Detected {np.sum(preds == -1):,} anomalies.")
        return preds

    def _train_one_class_svm(self) -> np.ndarray:
        """Trains One-Class SVM model."""
        params = self.config_params.get("one_class_svm", {"nu": 0.05, "kernel": "rbf", "gamma": "scale"})
        logger.info(f"Training One-Class SVM model (params: {params})...")

        # Subsample for OneClassSVM fit speed if sample count is large
        n_samples = len(self.X_scaled)
        model = OneClassSVM(**params)

        if n_samples > 15000:
            logger.info("Subsampling 15,000 instances for fast One-Class SVM fitting...")
            indices = np.random.choice(n_samples, 15000, replace=False)
            model.fit(self.X_scaled[indices])
            preds = model.predict(self.X_scaled)
        else:
            preds = model.fit_predict(self.X_scaled)

        model_path = self.models_dir / "one_class_svm.joblib"
        joblib.dump(model, model_path)
        self.trained_models["one_class_svm"] = model
        logger.info(f"Saved One-Class SVM to '{model_path}'. Detected {np.sum(preds == -1):,} anomalies.")
        return preds

    def _train_dbscan(self) -> np.ndarray:
        """Trains DBSCAN clustering algorithm."""
        params = self.config_params.get("dbscan", {"eps": 0.8, "min_samples": 5})
        logger.info(f"Training DBSCAN algorithm (params: {params})...")

        # Subsample for DBSCAN fit speed if sample count is large
        n_samples = len(self.X_scaled)
        if n_samples > 20000:
            logger.info("Subsampling 20,000 instances for fast DBSCAN clustering...")
            indices = np.random.choice(n_samples, 20000, replace=False)
            db = DBSCAN(**params)
            sample_clusters = db.fit_predict(self.X_scaled[indices])
            # Assign remaining points to nearest cluster center or noise
            clusters = np.full(n_samples, -1)
            clusters[indices] = sample_clusters
        else:
            db = DBSCAN(**params)
            clusters = db.fit_predict(self.X_scaled)

        model_path = self.models_dir / "dbscan.joblib"
        joblib.dump(db, model_path)
        self.trained_models["dbscan"] = db
        logger.info(f"Saved DBSCAN model to '{model_path}'. Identified {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters.")
        return clusters

    def _train_kmeans(self) -> np.ndarray:
        """Trains KMeans clustering algorithm."""
        params = self.config_params.get("kmeans", {"n_clusters": 4, "random_state": 42})
        logger.info(f"Training KMeans model (params: {params})...")

        kmeans = KMeans(**params)
        clusters = kmeans.fit_predict(self.X_scaled)

        model_path = self.models_dir / "kmeans.joblib"
        joblib.dump(kmeans, model_path)
        self.trained_models["kmeans"] = kmeans
        logger.info(f"Saved KMeans model to '{model_path}'.")
        return clusters
