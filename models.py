"""
models.py
Anomaly detection models for household circuit-level energy consumption.
Implements:
1. Isolation Forest (tree-based isolation of subspace outliers)
2. MLP Autoencoder (neural bottleneck reconstruction error anomaly detection)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

from features import extract_circuit_features, chronological_split, create_proxy_labels


class IsolationForestDetector:
    """
    Isolation Forest Anomaly Detector.
    Isolates anomalies by randomly selecting a feature and randomly selecting a split value.
    Anomalies require significantly fewer splits to isolate than normal instances.
    """
    def __init__(self, contamination=0.01, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def decision_function(self, X):
        """
        Returns continuous anomaly scores where higher value indicates greater anomaly likelihood.
        sklearn's decision_function returns negative values for outliers and positive for inliers.
        We invert the sign so higher score = higher anomaly.
        """
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        return -raw_scores

    def predict(self, X):
        """
        Returns binary predictions: 1 for anomaly, 0 for normal.
        """
        X_scaled = self.scaler.transform(X)
        raw_preds = self.model.predict(X_scaled)
        # raw_preds: -1 for outlier/anomaly, 1 for inlier/normal
        return (raw_preds == -1).astype(int)


class MLPAutoencoderDetector:
    """
    Multilayer Perceptron (MLP) Autoencoder Anomaly Detector.
    Uses a symmetrical neural bottleneck architecture to reconstruct normalized feature vectors.
    Samples with high reconstruction error (MSE) deviate from the learned normal manifold.
    Trained strictly on CPU within seconds using pure scikit-learn.
    """
    def __init__(self, hidden_layers=(16, 8, 16), threshold_percentile=99.0, random_state=42):
        self.hidden_layers = hidden_layers
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=0.01,
            max_iter=80,
            tol=1e-4,
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=5
        )
        self.threshold_ = None
        self.is_fitted = False

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, X_scaled)
        recon_train = self.model.predict(X_scaled)
        train_mse = np.mean(np.square(X_scaled - recon_train), axis=1)
        self.threshold_ = float(np.percentile(train_mse, self.threshold_percentile))
        self.is_fitted = True
        return self

    def decision_function(self, X):
        """
        Returns reconstruction error (MSE per sample) as continuous anomaly score.
        """
        X_scaled = self.scaler.transform(X)
        recon = self.model.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - recon), axis=1)
        return mse

    def predict(self, X, threshold=None):
        """
        Returns binary predictions: 1 for anomaly, 0 for normal.
        """
        mse = self.decision_function(X)
        cutoff = self.threshold_ if threshold is None else threshold
        return (mse > cutoff).astype(int)


def train_and_evaluate_circuit(df_hourly, circuit_col="Sub_metering_1", contamination=0.01, ae_percentile=99.0):
    """
    Runs end-to-end training and evaluation for a given circuit:
    - Extracts circuit features
    - Performs chronological 80/20 train/test split
    - Fits Isolation Forest and MLP Autoencoder on train split
    - Evaluates both detectors against proxy 99th percentile spike labels on test split
    - Returns metrics dictionary, trained models, predictions, scores, and test dataframe.
    """
    feats = extract_circuit_features(df_hourly, circuit_col=circuit_col)
    train_df, test_df = chronological_split(feats, train_ratio=0.8)

    feature_cols = [c for c in feats.columns]
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Proxy ground-truth labels on test set (top 1% consumption spikes)
    y_test_proxy, spike_cutoff = create_proxy_labels(test_df["consumption"], percentile=99.0)

    # 1. Train Isolation Forest
    iso = IsolationForestDetector(contamination=contamination, random_state=42)
    iso.fit(X_train)
    iso_preds = iso.predict(X_test)
    iso_scores = iso.decision_function(X_test)

    # 2. Train Autoencoder
    ae = MLPAutoencoderDetector(threshold_percentile=ae_percentile, random_state=42)
    ae.fit(X_train)
    ae_preds = ae.predict(X_test)
    ae_scores = ae.decision_function(X_test)

    # Metric computations
    def compute_metrics(y_true, y_pred, scores):
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = np.nan
        return {
            "Precision": float(p),
            "Recall": float(r),
            "F1-Score": float(f1),
            "Accuracy": float(acc),
            "ROC-AUC": float(auc),
            "Detected_Anomalies": int(y_pred.sum()),
            "True_Proxy_Anomalies": int(y_true.sum())
        }

    iso_metrics = compute_metrics(y_test_proxy, iso_preds, iso_scores)
    ae_metrics = compute_metrics(y_test_proxy, ae_preds, ae_scores)

    results = {
        "circuit": circuit_col,
        "test_df": test_df,
        "y_test_proxy": y_test_proxy,
        "spike_cutoff": spike_cutoff,
        "iso_model": iso,
        "iso_preds": iso_preds,
        "iso_scores": iso_scores,
        "iso_metrics": iso_metrics,
        "ae_model": ae,
        "ae_preds": ae_preds,
        "ae_scores": ae_scores,
        "ae_threshold": ae.threshold_,
        "ae_metrics": ae_metrics,
    }
    return results


if __name__ == "__main__":
    from data_loader import get_hourly_data
    df = get_hourly_data()
    for circuit in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        res = train_and_evaluate_circuit(df, circuit)
        print(f"\n--- {circuit} ---")
        print("Isolation Forest:", res["iso_metrics"])
        print("MLP Autoencoder: ", res["ae_metrics"])
