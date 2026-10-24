# Smart Energy Predictor: Circuit-Level Consumption Anomaly Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![UI-Gradio](https://img.shields.io/badge/UI-Gradio_6.30-orange.svg)](https://gradio.app/)
[![Dataset-UCI](https://img.shields.io/badge/Dataset-UCI_Household_Power-green.svg)](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
[![Architecture-CPU_Trainable](https://img.shields.io/badge/Compute-100%25_CPU_Trainable-brightgreen.svg)]()

An unsupervised anomaly detection system for sub-metering circuits in residential electrical systems. Built purely in Python, trained on 4 years of continuous smart meter telemetry from the official **UCI Household Electric Power Consumption** benchmark dataset, and served through a lightweight Gradio interface.

---

## 🎯 Project Scope & Honest Engineering Disclosures

This repository was originally titled with an overly broad ambition (*"predict energy usage for any city or country in the world"*). To adhere to rigorous machine learning and domain engineering standards, the project has been refocused into a concrete, verifiable technical scope:

1. **Single Household Focus**:
   - The underlying dataset captures electrical telemetry from **one private residence located in Sceaux, France** (near Paris) between December 2006 and November 2010.
   - It captures real appliance behaviors (dishwashers, washing machines, water heaters, air conditioning) specific to this household's lifestyle and European 230V infrastructure.
2. **No False Global Claims**:
   - Energy consumption patterns cannot be predicted or anomaly-detected zero-shot across "any city globally." Regional climate, local insulation standards, cooking fuel choices (gas vs electric), and heating systems differ drastically between geographies.
3. **No Synthetic Data**:
   - All models are trained and evaluated strictly against the **real ~2.07 million minute-level records** from the UCI repository.
4. **Pure Python Architecture**:
   - Zero JavaScript, zero React, and zero heavy node build tooling. The entire pipeline—from data streaming and feature extraction to neural autoencoding and interactive visual dashboards—runs natively in Python.

---

## 📁 Repository Structure

```text
SmartEnergyPredictor/
├── data/
│   ├── household_power_consumption.txt    # Official UCI minute-level dataset (~126.8 MB)
│   ├── household_power_consumption.zip    # Downloaded archive (~19.7 MB)
│   └── hourly_power_consumption.csv       # Resampled hourly dataset (34,589 rows)
├── models/
│   ├── sub_metering_1_iso.joblib          # Trained Isolation Forest (Kitchen)
│   ├── sub_metering_1_ae.joblib           # Trained MLP Autoencoder (Kitchen)
│   ├── sub_metering_2_iso.joblib          # Trained Isolation Forest (Laundry)
│   ├── sub_metering_2_ae.joblib           # Trained MLP Autoencoder (Laundry)
│   ├── sub_metering_3_iso.joblib          # Trained Isolation Forest (Climate & AC)
│   └── sub_metering_3_ae.joblib           # Trained MLP Autoencoder (Climate & AC)
├── results/
│   ├── metrics_comparison.csv             # Computed Precision, Recall, F1, ROC-AUC
│   ├── sub_metering_1_anomalies.png       # Kitchen anomaly time-series visualization
│   ├── sub_metering_2_anomalies.png       # Laundry anomaly time-series visualization
│   ├── sub_metering_3_anomalies.png       # Climate & AC anomaly time-series visualization
│   └── anomaly_score_distribution.png     # Anomaly score density distributions
├── .gitignore                             # Ignored caches, environments, and raw zips
├── download_data.py                       # Automated downloader for official UCI dataset
├── data_loader.py                         # Parsing, cleaning, and hourly resampling
├── features.py                            # Temporal & rolling statistics feature pipeline
├── models.py                              # Isolation Forest & MLP Autoencoder implementations
├── evaluate.py                            # 80/20 chronological split benchmark & plot generator
├── app.py                                 # Interactive Gradio dashboard
├── requirements.txt                       # Pinned dependencies
├── RESULTS.md                             # Full empirical results & limitations report
└── README.md                              # Main documentation
```

---

## ⚡ Data Pipeline & Engineering Decisions

### 1. The Official UCI Dataset
The dataset records 2,075,259 measurements sampled at 1-minute intervals from December 2006 to November 2010.
- **Sub_metering_1**: Kitchen circuit (mainly dishwasher, microwave, and electric oven).
- **Sub_metering_2**: Laundry circuit (washing machine, tumble dryer, refrigerator, lighting).
- **Sub_metering_3**: Climate control circuit (electric water-heater and air-conditioner).
- Format: Semicolon-separated values (`;`), with missing values marked as `?`.

### 2. Missing Value Imputation (`ffill + bfill`)
- Approximately 1.25% of timestamp entries contain missing values due to telemetry dropouts or meter disconnection.
- **Decision**: Missing entries are coerced to NaN and forward-filled (`ffill`), followed by backward-filling (`bfill`) for edge cases.
- **Rationale**: In high-frequency electrical metering, a momentary communication loss does not mean appliance power suddenly dropped to zero or jumped to the daily mean. Last observation carried forward preserves physical state continuity without introducing synthetic oscillations.

### 3. Hourly Mean Resampling (`resample('1h').mean()`)
- **Decision**: The minute-level time series is aggregated to 1-hour mean intervals, producing **34,589 hourly records**.
- **Rationale**:
  1. *Noise Suppression*: Sub-minute motor inrush currents and switching spikes are filtered, leaving clear operational regimes.
  2. *Appliance Cycle Modeling*: Dishwasher and laundry cycles operate over multi-hour periods. Hourly data captures equipment operational cycles and diurnal human schedules.
  3. *CPU Feasibility*: Reduces data volume by a factor of 60, allowing neural autoencoders and isolation forests to fit within seconds on standard laptop CPUs with zero GPU requirements.

---

## 🧠 Modeling & Algorithmic Design

### Circuit Focus
Anomaly detection is performed independently per sub-metering circuit:
- `Sub_metering_1` (Kitchen)
- `Sub_metering_2` (Laundry Room)
- `Sub_metering_3` (Climate & Heating)

### Feature Engineering
For each circuit, the feature pipeline generates 14 dimensions:
1. `consumption`: Active energy in watt-hours.
2. `hour`: Hour of day (0–23).
3. `weekday`: Day of week (0–6).
4. `is_weekend`: Binary flag (1 for Saturday/Sunday).
5. `hour_sin`, `hour_cos`: Cyclical sine/cosine transformation of hour.
6. `weekday_sin`, `weekday_cos`: Cyclical sine/cosine transformation of day.
7. `rolling_mean_6h`, `rolling_std_6h`: 6-hour moving window mean and standard deviation.
8. `rolling_mean_24h`, `rolling_std_24h`: 24-hour moving window mean and standard deviation.
9. `diff_from_6h_mean`, `diff_from_24h_mean`: Instantaneous deviation from local baselines.

### Two Algorithmic Detectors

| Model | Technique | Key Hyperparameters | Anomaly Score Definition |
| :--- | :--- | :--- | :--- |
| **Isolation Forest** | Non-parametric tree ensemble recursive splitting | `n_estimators=100`, `contamination=0.01`, `n_jobs=-1` | `-decision_function(X)` (Inverted path length; higher = isolated faster = anomalous) |
| **MLP Autoencoder** | Symmetrical neural bottleneck reconstruction | Architecture `[14 -> 16 -> 8 -> 16 -> 14]`, `Adam`, `max_iter=80`, `early_stopping=True` | Reconstruction Error $\text{MSE}_i = \frac{1}{D}\sum_{j=1}^D (x_{ij} - \hat{x}_{ij})^2$ |

#### Algorithmic Justifications:
- **Isolation Forest**: Ideal for detecting rare, high-magnitude consumption surges that deviate drastically from typical feature subspaces without assuming Gaussian distributions.
- **MLP Autoencoder**: Compresses 14-dimensional appliance context into an 8-dimensional latent representation. Learns normal diurnal relationships; departures from learned multivariate dynamics produce large reconstruction errors. Implemented in pure `scikit-learn` for rapid CPU training.

### Train / Test Split
- **Chronological 80/20 split** without shuffling:
  - **Train**: Dec 16, 2006 to Feb 11, 2010 (27,671 hourly samples).
  - **Test**: Feb 11, 2010 to Nov 26, 2010 (6,918 hourly samples).
- Chronological ordering is strictly preserved to prevent temporal leakage.

### Anomaly Thresholding & Proxy Ground-Truth
- **Ground-Truth Limitation**: The UCI dataset contains raw telemetry but no true equipment breakdown or fault labels.
- **Proxy Label**: As standard in unsupervised energy anomaly literature, the top 1% consumption spikes (99th percentile cutoff on the test set) serve as the evaluation proxy target.
- **Thresholds**:
  - Isolation Forest: Contamination set to 0.01 (1%).
  - Autoencoder: 99.0th percentile of training reconstruction error MSE.

---

## 📊 Empirical Results Summary

*Metrics computed directly on the 6,918-hour test split:*

| Circuit | Description | Model | Precision | Recall | F1-Score | Accuracy | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sub_metering_1** | Kitchen (dishwasher, oven, microwave) | **Isolation Forest** | **0.5185** | **0.4000** | **0.4516** | **0.9902** | **0.9946** |
| Sub_metering_1 | Kitchen (dishwasher, oven, microwave) | MLP Autoencoder | 0.1186 | 0.1000 | 0.1085 | 0.9834 | 0.9492 |
| **Sub_metering_2** | Laundry (washer, dryer, fridge) | **Isolation Forest** | **0.8500** | **0.4857** | **0.6182** | **0.9939** | **0.9970** |
| Sub_metering_2 | Laundry (washer, dryer, fridge) | MLP Autoencoder | 0.2188 | 0.1000 | 0.1373 | 0.9873 | 0.9086 |
| **Sub_metering_3** | Climate & Heating (water heater, AC) | **Isolation Forest** | **0.0455** | **0.0286** | **0.0351** | **0.9841** | **0.8409** |
| Sub_metering_3 | Climate & Heating (water heater, AC) | MLP Autoencoder | 0.0000 | 0.0000 | 0.0000 | 0.9799 | 0.4811 |

> For comprehensive graphs, ROC curves, and detailed circuit breakdown analysis, refer to [RESULTS.md](file:///RESULTS.md).

---

## 🚀 Quickstart & Reproduction Guide

### 1. Installation
Clone this repository and install the pinned dependencies:

```bash
git clone https://github.com/sanjnashenoy05-glitch/SmartEnergyPredictor.git
cd SmartEnergyPredictor
pip install -r requirements.txt
```

### 2. Download Real Dataset
Fetch and extract the official UCI dataset (~20MB compressed, ~126.8MB extracted):

```bash
python download_data.py
```
*(If already downloaded, the script detects the file and skips redownloading).*

### 3. Run Benchmark Evaluation & Regenerate Plots
Train the models on the 80% train split, evaluate on the 20% test split, and generate comparison artifacts:

```bash
python evaluate.py
```
This produces:
- `results/metrics_comparison.csv`
- `results/sub_metering_1_anomalies.png`
- `results/sub_metering_2_anomalies.png`
- `results/sub_metering_3_anomalies.png`
- `results/anomaly_score_distribution.png`
- `models/*.joblib`

### 4. Launch the Interactive Gradio UI
Launch the Python dashboard:

```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

#### UI Features:
- **Circuit Switcher**: Toggle between Kitchen, Laundry, and Climate/AC.
- **Model Selector**: Switch between Isolation Forest, Autoencoder, or Side-by-Side view.
- **Date Presets & Range**: Quickly view historical peak periods (e.g. Summer 2010 Peak) or enter custom dates.
- **Sensitivity Slider**: Adjust contamination cutoff dynamically from 0.1% to 5.0%.
- **Live Visuals**: Interactive time series with anomalous hours highlighted, anomaly score trajectories, and test metrics table.

---

## ⚠️ Known Limitations

1. **Univariate Proxy vs Multivariate Anomaly**:
   - The Autoencoder detects multidimensional relationship violations (e.g. moderate load running at 3:00 AM), which the univariate 99th percentile proxy spike target does not consider an anomaly. This explains the discrepancy in precision metrics between the two detectors.
2. **Climate Control Ceiling Effect**:
   - For `Sub_metering_3` (water heater & AC), appliance consumption is strictly capped by rated wattage (~21 Wh). Thermostatic heating cycles naturally reach near-maximum capacity, making standard spike detection unsuitable without external weather data.
3. **No Outdoor Temperature Telemetry**:
   - Heating and cooling demand is heavily driven by outdoor ambient temperatures. Without localized meteorological inputs, HVAC model predictions remain baseline estimates.
