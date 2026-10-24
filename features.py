"""
features.py
Feature engineering module for circuit-level consumption anomaly detection.
Generates temporal features, rolling-window statistics, and proxy ground-truth labels.
"""

import numpy as np
import pandas as pd

CIRCUIT_DESCRIPTIONS = {
    "Sub_metering_1": "Kitchen (dishwasher, microwave, electric oven)",
    "Sub_metering_2": "Laundry Room (washing machine, tumble dryer, refrigerator, light)",
    "Sub_metering_3": "Climate & Heating (electric water-heater and air-conditioner)"
}

FEATURE_COLUMNS = [
    "consumption",
    "hour",
    "weekday",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "rolling_mean_6h",
    "rolling_std_6h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "diff_from_6h_mean",
    "diff_from_24h_mean"
]


def extract_circuit_features(df, circuit_col="Sub_metering_1"):
    """
    Extracts time features, rolling statistics, and deviations for a given circuit.
    
    Parameters:
        df (pd.DataFrame): Hourly dataframe with DatetimeIndex.
        circuit_col (str): Circuit column name ('Sub_metering_1', 'Sub_metering_2', or 'Sub_metering_3').
        
    Returns:
        pd.DataFrame: Feature matrix indexed by Datetime.
    """
    if circuit_col not in df.columns:
        raise ValueError(f"Circuit '{circuit_col}' not found in DataFrame columns: {list(df.columns)}")

    feat_df = pd.DataFrame(index=df.index)

    # Base circuit consumption in watt-hours
    feat_df["consumption"] = df[circuit_col].astype(float)

    # Time features
    feat_df["hour"] = df.index.hour
    feat_df["weekday"] = df.index.weekday  # 0 = Monday, 6 = Sunday
    feat_df["is_weekend"] = (df.index.weekday >= 5).astype(int)

    # Cyclical hour encoding (24h period)
    feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24.0)
    feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24.0)

    # Cyclical weekday encoding (7d period)
    feat_df["weekday_sin"] = np.sin(2 * np.pi * feat_df["weekday"] / 7.0)
    feat_df["weekday_cos"] = np.cos(2 * np.pi * feat_df["weekday"] / 7.0)

    # Rolling window statistics (6-hour and 24-hour windows)
    feat_df["rolling_mean_6h"] = feat_df["consumption"].rolling(window=6, min_periods=1).mean()
    feat_df["rolling_std_6h"] = feat_df["consumption"].rolling(window=6, min_periods=1).std().fillna(0.0)

    feat_df["rolling_mean_24h"] = feat_df["consumption"].rolling(window=24, min_periods=1).mean()
    feat_df["rolling_std_24h"] = feat_df["consumption"].rolling(window=24, min_periods=1).std().fillna(0.0)

    # Rate of departure from local baselines
    feat_df["diff_from_6h_mean"] = feat_df["consumption"] - feat_df["rolling_mean_6h"]
    feat_df["diff_from_24h_mean"] = feat_df["consumption"] - feat_df["rolling_mean_24h"]

    return feat_df


def create_proxy_labels(series, threshold=None, percentile=99.0):
    """
    Creates proxy anomaly labels (top 1% consumption spikes).
    
    Limitation Note:
    The UCI dataset does not provide labeled ground-truth equipment fault or leakage events.
    In commercial anomaly detection literature, extreme statistical consumption spikes
    (top 1% highest hourly usage for that circuit) serve as the benchmark proxy target.
    
    Parameters:
        series (pd.Series or np.ndarray): Consumption values.
        threshold (float, optional): Explicit cutoff value. If None, derived from percentile.
        percentile (float): Percentile cutoff (default 99.0 = top 1%).
        
    Returns:
        np.ndarray: Binary 0/1 array (1 = proxy anomaly, 0 = normal).
        float: Computed threshold value.
    """
    vals = np.asarray(series)
    if threshold is None:
        threshold = float(np.percentile(vals, percentile))
    y_proxy = (vals > threshold).astype(int)
    return y_proxy, threshold


def chronological_split(df_features, train_ratio=0.8):
    """
    Chronological train/test split (no shuffling to prevent data leakage in time series).
    First 80% train, last 20% test.
    """
    n_total = len(df_features)
    split_idx = int(n_total * train_ratio)

    df_train = df_features.iloc[:split_idx].copy()
    df_test = df_features.iloc[split_idx:].copy()

    return df_train, df_test


if __name__ == "__main__":
    from data_loader import get_hourly_data
    df = get_hourly_data()
    feats = extract_circuit_features(df, "Sub_metering_1")
    train_df, test_df = chronological_split(feats)
    print(f"Features shape: {feats.shape}")
    print(f"Train split: {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Test split:  {len(test_df)} rows ({test_df.index[0]} to {test_df.index[-1]})")
    y_test_proxy, thresh = create_proxy_labels(test_df["consumption"], percentile=99.0)
    print(f"Test proxy threshold (99th pct): {thresh:.2f} Wh | Anomaly count: {y_test_proxy.sum()} ({y_test_proxy.mean()*100:.2f}%)")
