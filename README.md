# Global Smart Energy Predictor

This project predicts short-term household energy consumption for any city using real-time weather data from the OpenWeather API and a machine learning model trained on the UCI Energy Dataset.  
It helps users estimate energy usage trends and understand how temperature and humidity influence power consumption.

## Features
- Fetches live weather data for any city.
- Predicts next 6 hours of energy consumption using a trained Random Forest model.
- Displays results with a detailed time-series graph and insights.
- Built with Python, Pandas, Scikit-learn, Plotly, and Gradio.

## Usage
1. Enter a city name (e.g., **Chennai**, **Seoul**, **London**) in the input box.  
2. The system fetches live temperature and humidity for that city.  
3. The ML model predicts energy consumption for the next 6 hours and plots a graph.  
4. The app provides a short interpretation of the results.

## Tech Stack
- Python 3.12  
- Gradio for interface  
- Scikit-learn for ML  
- OpenWeather API for real-time data  
- Plotly for visualization

## How to Run
```bash
pip install -r requirements.txt
python app.py
