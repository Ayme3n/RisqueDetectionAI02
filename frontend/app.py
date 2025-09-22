import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

# Backend URL (adjust if needed)
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("🔍 AI-Powered Anomaly Detection")

st.sidebar.header("Upload Logs")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Logs", df.head())

    if st.sidebar.button("🚀 Run Prediction"):
        try:
            # sanitize: replace inf with None and NaN with None
            df_sanitized = df.replace([np.inf, -np.inf], None).where(pd.notnull(df), None)

            # ensure python-native objects (optional but sometimes helpful)
            df_sanitized = df_sanitized.astype(object)

            # prepare records for JSON -- uses the "records" field expected by your API
            payload = {"records": df_sanitized.to_dict(orient="records")}

            response = requests.post(API_URL, json=payload, timeout=60)

            if response.status_code == 200:
                results = response.json()["predictions"]
                results_df = pd.DataFrame(results)
                st.success("✅ Predictions received!")
                st.dataframe(results_df.head(50))
                # ... plotting etc.
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Request failed: {e}")
