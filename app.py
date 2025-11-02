import gradio as gr
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import plotly.express as px

API_KEY = "78994cc63fa1018861ff9684abfc1531"  # Replace this
MODEL_PATH = "energy_forecast_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except:
    model = None

def get_weather(city):
    """Fetch live weather data for the given city."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url)
        data = res.json()
        if res.status_code != 200 or "main" not in data:
            return None, None, None, None, f"Could not fetch weather for '{city}'."
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        tz_offset = data.get("timezone", 0)
        tz = pytz.FixedOffset(tz_offset / 60)
        local_time = datetime.now(timezone.utc).astimezone(tz)
        return temp, humidity, tz, local_time, None
    except Exception as e:
        return None, None, None, None, f"Error fetching weather: {e}"

def predict_energy(city):
    """Predict next 6 hours of energy consumption based on current weather."""
    if not model:
        return "Model not found. Please ensure 'energy_forecast_model.joblib' is uploaded.", None, None

    temp, humidity, tz, local_time, error = get_weather(city)
    if error:
        return error, None, None

    # Simulate voltage & intensity variations based on weather
    base_voltage = 230 + (humidity - 50) * 0.05
    base_intensity = 2 + (temp - 25) * 0.1

    preds, timestamps = [], []
    for i in range(1, 7):
        X = pd.DataFrame({
            "voltage": [base_voltage + np.random.uniform(-2, 2)],
            "intensity": [base_intensity + np.random.uniform(-0.3, 0.3)]
        })
        preds.append(model.predict(X)[0])
        timestamps.append(local_time + timedelta(hours=i))

    forecast_df = pd.DataFrame({"timestamp": timestamps, "predicted_kwh": preds})
    fig = px.line(forecast_df, x="timestamp", y="predicted_kwh", markers=True,
                  title=f"Predicted Energy Consumption in {city.title()} (Next 6 Hours)",
                  labels={"timestamp": "Local Time", "predicted_kwh": "Predicted Energy (kWh)"})
    q1, q2, q3 = df['consumption_kwh'].quantile([0.33, 0.66, 0.9])
    pred=np.mean(preds)
    if pred < q1:
      level = "Low"
    elif pred < q2:
      level = "Moderate"
    else:
      level = "High"
    return msg, forecast_df, fig

app = gr.Interface(
    fn=predict_energy,
    inputs=gr.Textbox(label="Enter City", placeholder="e.g. Chennai, Tokyo, New York"),
    outputs=[
        gr.Textbox(label="Message / Inference"),
        gr.Dataframe(label="Predicted Energy (kWh)"),
        gr.Plot(label="Energy Forecast Plot")
    ],
    title="Global Smart Energy Forecast",
    description="Predicts short-term household energy consumption using live weather data for any city."
)

app.launch(share=True)
