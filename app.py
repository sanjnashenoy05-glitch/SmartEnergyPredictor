"""
app.py
Interactive Gradio application for circuit-level consumption anomaly detection.
Features:
- Circuit selection: Sub_metering_1 (Kitchen), Sub_metering_2 (Laundry), Sub_metering_3 (Climate/AC)
- Model selection: Isolation Forest, MLP Autoencoder, or Both
- Date range selection with quick presets
- Interactive sensitivity / contamination threshold slider
- Time-series plot with highlighted detected anomalies
- Anomaly score and reconstruction error trajectory
- Quantitative metrics comparison table
- Complete documentation of scope and limitations
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Thread-safe headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gradio as gr

from data_loader import get_hourly_data
from features import extract_circuit_features, chronological_split, create_proxy_labels, CIRCUIT_DESCRIPTIONS
from models import IsolationForestDetector, MLPAutoencoderDetector

# Global cache for data and pre-trained models
DATA_CACHE = None
MODEL_CACHE = {}
METRICS_CACHE = None
MODELS_DIR = "models"
RESULTS_DIR = "results"


def initialize_app_data(data_dir="data", results_dir=RESULTS_DIR, models_dir=MODELS_DIR):
    """Loads hourly data and pre-trained models for instant sub-second UI responsiveness."""
    global DATA_CACHE, MODEL_CACHE, METRICS_CACHE

    if DATA_CACHE is not None and MODEL_CACHE:
        return

    print("Initializing application data...", flush=True)
    DATA_CACHE = get_hourly_data(data_dir=data_dir)

    metrics_csv = os.path.join(results_dir, "metrics_comparison.csv")
    if os.path.exists(metrics_csv):
        METRICS_CACHE = pd.read_csv(metrics_csv)
    else:
        METRICS_CACHE = None

    print("Loading or fitting models for Sub_metering circuits...", flush=True)
    for circuit in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        feats = extract_circuit_features(DATA_CACHE, circuit_col=circuit)
        train_df, test_df = chronological_split(feats, train_ratio=0.8)

        iso_path = os.path.join(models_dir, f"{circuit.lower()}_iso.joblib")
        ae_path = os.path.join(models_dir, f"{circuit.lower()}_ae.joblib")

        # Load cached models if available, else fit on train split
        if os.path.exists(iso_path) and os.path.exists(ae_path):
            iso = joblib.load(iso_path)
            ae = joblib.load(ae_path)
        else:
            feature_cols = [c for c in feats.columns]
            X_train = train_df[feature_cols].values
            iso = IsolationForestDetector(contamination=0.01, random_state=42)
            iso.fit(X_train)
            ae = MLPAutoencoderDetector(threshold_percentile=99.0, random_state=42)
            ae.fit(X_train)

        # Baseline spike threshold (99th pct of full dataset)
        _, spike_thresh = create_proxy_labels(DATA_CACHE[circuit], percentile=99.0)

        MODEL_CACHE[circuit] = {
            "iso": iso,
            "ae": ae,
            "feats": feats,
            "spike_thresh": spike_thresh,
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0]
        }
    print("Application data and models ready.", flush=True)


def analyze_circuit(circuit_key, model_choice, date_preset, start_date_str, end_date_str, contamination_pct):
    """
    Main callback invoked by Gradio controls.
    """
    if DATA_CACHE is None or not MODEL_CACHE:
        initialize_app_data()

    # Map friendly display names
    circuit_col = "Sub_metering_1"
    if "Laundry" in circuit_key or "Sub_metering_2" in circuit_key:
        circuit_col = "Sub_metering_2"
    elif "Climate" in circuit_key or "Sub_metering_3" in circuit_key:
        circuit_col = "Sub_metering_3"

    cache = MODEL_CACHE[circuit_col]
    feats = cache["feats"]
    spike_thresh = cache["spike_thresh"]

    # Resolve date range
    if date_preset == "Test Set - June 2010 (Summer Peak)":
        start_dt = pd.Timestamp("2010-06-01 00:00:00")
        end_dt = pd.Timestamp("2010-06-30 23:00:00")
    elif date_preset == "Test Set - Full Test Period (Feb to Nov 2010)":
        start_dt = pd.Timestamp("2010-02-12 00:00:00")
        end_dt = pd.Timestamp("2010-11-26 21:00:00")
    elif date_preset == "Train Set - July 2009 (Historical Summer)":
        start_dt = pd.Timestamp("2009-07-01 00:00:00")
        end_dt = pd.Timestamp("2009-07-31 23:00:00")
    elif date_preset == "Train Set - Dec 2008 (Winter Peak)":
        start_dt = pd.Timestamp("2008-12-01 00:00:00")
        end_dt = pd.Timestamp("2008-12-31 23:00:00")
    else:
        try:
            start_dt = pd.Timestamp(start_date_str)
            end_dt = pd.Timestamp(end_date_str)
        except Exception:
            start_dt = pd.Timestamp("2010-06-01")
            end_dt = pd.Timestamp("2010-06-30")

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    # Slice features
    sliced_feats = feats.loc[start_dt:end_dt]
    if len(sliced_feats) == 0:
        sliced_feats = feats.loc["2010-06-01":"2010-06-30"]
        start_dt, end_dt = sliced_feats.index[0], sliced_feats.index[-1]

    idx = sliced_feats.index
    consumption = sliced_feats["consumption"].values
    X_slice = sliced_feats.values

    # Model inference
    iso_model = cache["iso"]
    ae_model = cache["ae"]

    iso_scores = iso_model.decision_function(X_slice)
    ae_scores = ae_model.decision_function(X_slice)

    # Dynamic thresholding based on user sensitivity slider
    contamination_ratio = max(0.001, min(0.20, contamination_pct / 100.0))
    iso_cutoff = np.percentile(iso_scores, 100.0 * (1.0 - contamination_ratio))
    ae_cutoff = np.percentile(ae_scores, 100.0 * (1.0 - contamination_ratio))

    iso_anomalies = (iso_scores > iso_cutoff).astype(int)
    ae_anomalies = (ae_scores > ae_cutoff).astype(int)

    # Summary KPIs
    n_hours = len(sliced_feats)
    peak_wh = float(np.max(consumption)) if n_hours > 0 else 0.0

    if "Isolation Forest" in model_choice:
        n_anom = int(iso_anomalies.sum())
    elif "Autoencoder" in model_choice:
        n_anom = int(ae_anomalies.sum())
    else:
        n_anom = int((iso_anomalies | ae_anomalies).sum())

    anom_pct = (n_anom / n_hours * 100.0) if n_hours > 0 else 0.0

    summary_text = (
        f"### Analysis Summary\n"
        f"- **Circuit**: `{circuit_col}` ({CIRCUIT_DESCRIPTIONS.get(circuit_col)})\n"
        f"- **Time Window**: `{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}` ({n_hours:,} hourly readings)\n"
        f"- **Model Selected**: `{model_choice}`\n"
        f"- **Detected Anomalies**: **{n_anom:,}** hours ({anom_pct:.2f}% of window)\n"
        f"- **Peak Consumption in Range**: **{peak_wh:.2f} Wh** (Historical 99th Pct Baseline: `{spike_thresh:.2f} Wh`)\n"
    )

    # ================= PLOT 1: Consumption Time Series =================
    fig1, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.plot(idx, consumption, color="#1f77b4", lw=1.3, label="Hourly Consumption (Wh)", alpha=0.85)
    ax1.axhline(spike_thresh, color="#d9534f", linestyle="--", lw=1.2, label=f"Proxy Threshold ({spike_thresh:.1f} Wh)")

    if model_choice in ["Isolation Forest", "Both (Side-by-Side Comparison)"]:
        mask_iso = (iso_anomalies == 1)
        if mask_iso.any():
            ax1.scatter(idx[mask_iso], consumption[mask_iso], color="#e74c3c", marker="o", s=45,
                        label="Isolation Forest Anomaly", zorder=5, edgecolors="black", linewidths=0.6)

    if model_choice in ["MLP Autoencoder", "Both (Side-by-Side Comparison)"]:
        mask_ae = (ae_anomalies == 1)
        if mask_ae.any():
            ax1.scatter(idx[mask_ae], consumption[mask_ae], color="#f39c12", marker="^", s=50,
                        label="Autoencoder Anomaly", zorder=6, edgecolors="black", linewidths=0.6)

    ax1.set_title(f"Circuit Energy Consumption & Detected Anomalies: {circuit_col}\n({CIRCUIT_DESCRIPTIONS.get(circuit_col)})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Active Energy (Wh)", fontsize=10)
    ax1.set_xlabel("Time", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:00"))
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    # ================= PLOT 2: Anomaly Scores =================
    fig2, ax2 = plt.subplots(figsize=(12, 3.2))

    def min_max(s):
        s_min, s_max = np.min(s), np.max(s)
        return (s - s_min) / (s_max - s_min + 1e-9)

    if model_choice in ["Isolation Forest", "Both (Side-by-Side Comparison)"]:
        ax2.plot(idx, min_max(iso_scores), color="#e74c3c", lw=1.1, label="Isolation Forest Score (Norm)")
        ax2.axhline(min_max(np.array([iso_cutoff]))[0], color="#e74c3c", linestyle=":", lw=1.0, alpha=0.7)

    if model_choice in ["MLP Autoencoder", "Both (Side-by-Side Comparison)"]:
        ax2.plot(idx, min_max(ae_scores), color="#f39c12", lw=1.1, label="Autoencoder Recon MSE (Norm)")
        ax2.axhline(min_max(np.array([ae_cutoff]))[0], color="#f39c12", linestyle=":", lw=1.0, alpha=0.7)

    ax2.set_title("Normalized Anomaly Scores Over Time", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Score [0, 1]", fontsize=9)
    ax2.set_xlabel("Time", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:00"))
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8)
    plt.tight_layout()

    # Filter metrics table for selected circuit
    if METRICS_CACHE is not None:
        table_df = METRICS_CACHE[METRICS_CACHE["Circuit"] == circuit_col].copy()
    else:
        table_df = pd.DataFrame()

    return summary_text, fig1, fig2, table_df


def update_date_inputs(preset_choice):
    """Sync textboxes when preset dropdown changes."""
    presets = {
        "Test Set - June 2010 (Summer Peak)": ("2010-06-01", "2010-06-30"),
        "Test Set - Full Test Period (Feb to Nov 2010)": ("2010-02-12", "2010-11-26"),
        "Train Set - July 2009 (Historical Summer)": ("2009-07-01", "2009-07-31"),
        "Train Set - Dec 2008 (Winter Peak)": ("2008-12-01", "2008-12-31"),
    }
    if preset_choice in presets:
        return presets[preset_choice][0], presets[preset_choice][1]
    return "2010-06-01", "2010-06-30"


def create_ui():
    """Builds the Gradio Blocks UI."""
    with gr.Blocks(title="Smart Energy Predictor: Circuit Anomaly Detection") as demo:
        gr.Markdown(
            """
            # Smart Energy Predictor: Circuit-Level Consumption Anomaly Detection
            **Unsupervised Anomaly Detection on Residential Sub-Metering Circuits using the UCI Benchmark Dataset**
            
            *Scope Notice*: Model trained on the real **UCI Individual Household Electric Power Consumption** dataset 
            (monitored in Sceaux, France, 2006–2010). It evaluates circuit-level appliance dynamics for this single household 
            and explicitly **does not claim** zero-shot generalization to 'any city or country globally'.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                circuit_radio = gr.Radio(
                    choices=[
                        "Sub_metering_1 (Kitchen)",
                        "Sub_metering_2 (Laundry)",
                        "Sub_metering_3 (Climate & AC)"
                    ],
                    value="Sub_metering_1 (Kitchen)",
                    label="1. Select Sub-Metering Circuit"
                )

                model_radio = gr.Radio(
                    choices=[
                        "Isolation Forest",
                        "MLP Autoencoder",
                        "Both (Side-by-Side Comparison)"
                    ],
                    value="Both (Side-by-Side Comparison)",
                    label="2. Select Anomaly Detector"
                )

                date_preset_dd = gr.Dropdown(
                    choices=[
                        "Test Set - June 2010 (Summer Peak)",
                        "Test Set - Full Test Period (Feb to Nov 2010)",
                        "Train Set - July 2009 (Historical Summer)",
                        "Train Set - Dec 2008 (Winter Peak)",
                        "Custom Date Range"
                    ],
                    value="Test Set - June 2010 (Summer Peak)",
                    label="3. Date Range Preset"
                )

                with gr.Row():
                    start_date_in = gr.Textbox(value="2010-06-01", label="Start Date (YYYY-MM-DD)")
                    end_date_in = gr.Textbox(value="2010-06-30", label="End Date (YYYY-MM-DD)")

                contamination_sl = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=1.0,
                    step=0.1,
                    label="4. Contamination / Sensitivity Threshold (%)"
                )

                submit_btn = gr.Button("Run Anomaly Detection", variant="primary")

            with gr.Column(scale=2):
                summary_md = gr.Markdown()
                plot_timeseries = gr.Plot(label="Time-Series Anomaly Visualization")
                plot_scores = gr.Plot(label="Anomaly Score Trajectory")
                metrics_df = gr.DataFrame(label="Benchmark Metrics on Test Split (Proxy Spike Target)")

        with gr.Accordion("Circuit Details, Methodology & Limitations", open=False):
            gr.Markdown(
                """
                ### Sub-Metering Circuit Definitions (UCI Dataset)
                - **Sub_metering_1**: Kitchen circuit powering dishwasher, microwave oven, and electric oven.
                - **Sub_metering_2**: Laundry room powering washing machine, tumble dryer, refrigerator, and light.
                - **Sub_metering_3**: Climate control powering electric water-heater and air conditioning.
                
                ### Algorithmic Comparison
                - **Isolation Forest**: Randomly isolates data points in hyperplanes. Outliers (large power surges) are isolated near the root of trees. Highly effective for sharp transient loads.
                - **MLP Autoencoder**: Neural compression bottleneck `[14 -> 16 -> 8 -> 16 -> 14]`. High reconstruction MSE indicates patterns deviating from typical cyclic correlations (e.g., unexpected appliances running overnight).
                
                ### Ground-Truth & Metric Limitations
                The UCI dataset provides continuous electrical telemetry but no ground-truth appliance breakdown logs.
                Therefore, evaluation metrics (Precision, Recall, F1) are benchmarked against **top-1% statistical consumption spikes** as a proxy target.
                High recall indicates the detector captured unusual high-load regimes; false positives in this context often represent subtle multi-variable shifts rather than noise.
                """
            )

        # Wire events
        date_preset_dd.change(
            fn=update_date_inputs,
            inputs=[date_preset_dd],
            outputs=[start_date_in, end_date_in]
        )

        submit_btn.click(
            fn=analyze_circuit,
            inputs=[circuit_radio, model_radio, date_preset_dd, start_date_in, end_date_in, contamination_sl],
            outputs=[summary_md, plot_timeseries, plot_scores, metrics_df]
        )

        # Run on load
        demo.load(
            fn=analyze_circuit,
            inputs=[circuit_radio, model_radio, date_preset_dd, start_date_in, end_date_in, contamination_sl],
            outputs=[summary_md, plot_timeseries, plot_scores, metrics_df]
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch Smart Energy Predictor Anomaly Detection Gradio UI")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default 7860)")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    args = parser.parse_args()

    # Pre-load data and cached models
    initialize_app_data()

    app = create_ui()
    print(f"Launching Gradio app on port {args.port}...", flush=True)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
