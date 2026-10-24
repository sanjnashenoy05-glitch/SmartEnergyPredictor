"""
data_loader.py
Module for loading, parsing, cleaning, and resampling the UCI Household Electric Power Consumption dataset.
"""

import os
import sys
import pandas as pd
from download_data import download_dataset, EXPECTED_FILENAME

RAW_FILENAME = EXPECTED_FILENAME
HOURLY_CACHE_FILENAME = "hourly_power_consumption.csv"

NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]


def load_raw_data(data_dir="data", filepath=None):
    """
    Loads the official UCI Household Electric Power Consumption dataset (~2M minute-level rows).
    - Parses Date and Time into a single datetime index.
    - Converts numeric columns to float, coercing '?' to NaN.
    - Forward-fills missing values (and backward-fills any leading edge cases).
    
    Documentation of Missing Value Strategy:
    In continuous minute-level smart meter monitoring, missing values typically reflect
    brief network telemetry dropouts or temporary sensor disconnects. Forward-filling
    (last observation carried forward) preserves the physical operational state of the household
    without inventing synthetic trends or disrupting time-series continuity.
    """
    if filepath is None:
        filepath = os.path.join(data_dir, RAW_FILENAME)

    if not os.path.exists(filepath):
        print(f"Dataset file '{filepath}' not found. Initiating download...")
        download_dataset(data_dir=data_dir)

    print(f"Loading raw dataset from {filepath}...")
    df = pd.read_csv(
        filepath,
        sep=';',
        na_values=['?'],
        low_memory=False
    )
    print(f"Loaded {len(df):,} raw rows.")

    print("Parsing datetime index...")
    # Date is DD/MM/YYYY, Time is HH:MM:SS
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)

    print("Coercing numeric columns and imputing missing values...")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Forward fill missing values, then backward fill any leading NaNs
    df = df.ffill().bfill()
    print("Raw dataset processing complete.")
    return df


def resample_hourly(df, cache_path=None):
    """
    Resamples minute-level data to hourly means.
    
    Documentation of Resampling Decision:
    1. Noise Reduction: Minute-level sub-metering data contains high-frequency transient switching
       noise (e.g. initial motor startup surges) that obscure sustained equipment anomaly patterns.
    2. Appliance Cycle Modeling: Major household circuits (kitchen dishwashers, laundry cycles,
       water heaters, air conditioning) operate on multi-minute to multi-hour cycles. Hourly mean
       aggregation captures meaningful operational regimes and human behavioral schedules.
    3. CPU Efficiency: Minute-level data comprises ~2,075,259 records. Aggregating to 1-hour
       intervals yields 34,589 records, perfectly suited for rapid, deterministic training
       on standard laptop CPUs in seconds without GPU requirements.
    """
    print("Resampling time series to hourly means...")
    df_hourly = df.resample('1h').mean()
    # Fill any potential resampling gap if present
    df_hourly = df_hourly.ffill().bfill()
    print(f"Resampled to {len(df_hourly):,} hourly records.")

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        print(f"Caching hourly dataset to {cache_path}...")
        df_hourly.to_csv(cache_path)
        print("Hourly cache saved.")

    return df_hourly


def get_hourly_data(data_dir="data", force_reload=False):
    """
    Convenience loader that loads cached hourly data if available,
    otherwise loads raw UCI data, resamples to hourly, and caches it.
    """
    cache_path = os.path.join(data_dir, HOURLY_CACHE_FILENAME)
    raw_path = os.path.join(data_dir, RAW_FILENAME)

    if os.path.exists(cache_path) and not force_reload:
        print(f"Loading cached hourly data from {cache_path}...")
        df_hourly = pd.read_csv(cache_path, index_col='Datetime', parse_dates=True)
        # Ensure all numeric columns are float
        for col in NUMERIC_COLUMNS:
            if col in df_hourly.columns:
                df_hourly[col] = pd.to_numeric(df_hourly[col], errors='coerce')
        return df_hourly

    df_raw = load_raw_data(data_dir=data_dir, filepath=raw_path)
    df_hourly = resample_hourly(df_raw, cache_path=cache_path)
    return df_hourly


if __name__ == "__main__":
    df = get_hourly_data(force_reload=False)
    print("\nHourly Data Sample:")
    print(df.head())
    print("\nSub-metering Summary Statistics:")
    print(df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].describe())
