# frontend/app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import json
import plotly.express as px
from pathlib import Path
from datetime import datetime
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.app import inference


USE_LOCAL_INFERENCE = False  # force API mode

# ---------------- CONFIG ----------------
API_URL = st.secrets.get("api_url", "http://127.0.0.1:8000/predict")  # change if your API is remote
API_FILE_URL = st.secrets.get("api_file_url", "http://127.0.0.1:8000/predict_file")  # optional endpoint
USE_API_BY_DEFAULT = True  # using API separation is recommended

st.set_page_config(page_title="RisqueDetectionAI — Dashboard", layout="wide")

# ---------------- Helpers ----------------
def sanitize_df_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf with None and NaN with None, convert numpy scalars to python scalars.
    Returns a new DataFrame safe to turn into JSON via json.dumps.
    """
    df2 = df.copy(deep=True)
    # Replace inf, -inf with NaN, then NaN -> None
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    df2 = df2.where(pd.notnull(df2), None)

    # Convert numpy scalar values (e.g. np.int64) to python types
    for col in df2.columns:
        df2[col] = df2[col].apply(lambda x: x.item() if isinstance(x, (np.generic,)) else x)
    return df2

def try_call_api_json(df: pd.DataFrame, api_url: str = API_URL, timeout=120):
    """
    Send JSON to API_URL (expects {"records": [...]}).
    Returns (success_bool, response_json_or_error_str).
    """
    payload = {"records": sanitize_df_for_json(df).to_dict(orient="records")}
    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
    except Exception as e:
        return False, f"Request failed: {e}"
    if resp.status_code != 200:
        return False, f"API error {resp.status_code}: {resp.text}"
    try:
        return True, resp.json()
    except Exception as e:
        return False, f"Failed to parse API JSON: {e}"

def try_call_api_file(uploaded_bytes: bytes, filename: str, api_file_url: str = API_FILE_URL, timeout=300):
    """
    Send multipart/form-data file upload to API_FILE_URL (if supported).
    Returns (success_bool, response_json_or_error_str).
    """
    try:
        files = {"file": (filename, uploaded_bytes, "text/csv")}
        resp = requests.post(api_file_url, files=files, timeout=timeout)
    except Exception as e:
        return False, f"File upload failed: {e}"
    if resp.status_code != 200:
        return False, f"API file error {resp.status_code}: {resp.text}"
    try:
        return True, resp.json()
    except Exception as e:
        return False, f"Failed to parse API JSON: {e}"

def local_predict(df: pd.DataFrame):
    """
    Try to import and call backend.app.inference.predict_batch locally.
    Use this only if the API is unavailable and you run streamlit in the same environment.
    """
    try:
        # Add package root to sys.path if needed elsewhere
        from backend.app import inference
    except Exception as e:
        return False, f"Local inference import failed: {e}"
    try:
        inference.load_artifacts()
        res = inference.predict_batch(df)
        return True, {"predictions": res.to_dict(orient="records")}
    except Exception as e:
        return False, f"Local inference run failed: {e}"

def parse_timestamp_column(df: pd.DataFrame):
    if "timestamp" in df.columns:
        try:
            df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            df["timestamp_parsed"] = None
    else:
        df["timestamp_parsed"] = None
    return df

# ---------------- UI ----------------
st.title("🔍 RisqueDetectionAI — Anomaly Detection Dashboard")
st.markdown(
    "Upload your structured Wazuh logs (CSV). This dashboard calls the inference backend "
    "and displays prioritized anomalies with visualizations and exports."
)

# Sidebar: upload and settings
with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload CSV (structured logs)", type=["csv"], help="Use cleaned_structured_wazuh_logs.csv or similar.")
    use_api = st.checkbox("Use backend API (preferred)", value=True)
    prefer_file_upload = st.checkbox("Try file upload endpoint (multipart)", value=False)
    st.markdown("---")
    st.header("Threshold & Filters")
    manual_threshold = st.slider("Local ensemble threshold (override)", min_value=0.0, max_value=1.0, value=0.7007, step=0.001)
    st.caption("You can experiment with threshold — this won't change the backend model, it only affects the view.")
    st.markdown("---")
    st.header("Quick actions")
    sample_btn = st.button("Load small sample")

# Load sample data (tiny) if requested
SAMPLE_CSV = """timestamp,agent.name,rule.mitre.id,rule.mitre.tactic,rule.level,rule.id
2024-07-01 01:00:00,agent-100,T1003,Credential Access,5,1001
2024-07-01 02:34:00,agent-101,T1059,Execution,7,1002
2024-07-02 14:00:00,agent-100,T1086,Execution,3,1003
"""
if sample_btn:
    uploaded_file = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    uploaded_file.name = "sample_small.csv"

# If a file is uploaded, read it as DataFrame
df_input = None
if uploaded_file is not None:
    try:
        # If uploaded_file is an UploadedFile object or BytesIO
        uploaded_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        df_input = pd.read_csv(io.BytesIO(uploaded_bytes))
        st.success(f"Loaded {len(df_input)} rows")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# Buttons to trigger prediction
if df_input is not None:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Run predictions (send JSON)"):
            st.session_state["_run_prediction"] = ("json", df_input)
    with col2:
        if st.button("Run predictions (file upload)"):
            st.session_state["_run_prediction"] = ("file", uploaded_bytes, getattr(uploaded_file, "name", "upload.csv"))
    with col3:
        st.write("Rows:", len(df_input))

# Run prediction when requested
predictions_df = None
last_api_msg = None
if "_run_prediction" in st.session_state:
    mode = st.session_state["_run_prediction"][0]
    st.info("Running inference... this may take a moment for large files.")
    if mode == "json":
        ok, resp = try_call_api_json(df_input) if use_api else (False, "API disabled by user")
        if ok:
            preds = resp.get("predictions") or resp.get("preds") or resp
            predictions_df = pd.DataFrame(preds)
            last_api_msg = "Backend API (JSON) succeeded."
        else:
            last_api_msg = str(resp)
            # fallback: try local prediction
            ok_local, resp_local = local_predict(df_input)
            if ok_local:
                predictions_df = pd.DataFrame(resp_local["predictions"])
                last_api_msg += " Local inference succeeded after API failure."
            else:
                last_api_msg += " Local inference failed: " + str(resp_local)
    else:  # file mode
        _, uploaded_bytes, filename = st.session_state["_run_prediction"]
        ok, resp = try_call_api_file(uploaded_bytes, filename) if (use_api and prefer_file_upload) else (False, "File upload not enabled or API disabled")
        if ok:
            preds = resp.get("predictions") or resp
            predictions_df = pd.DataFrame(preds.get("predictions") if "predictions" in preds else preds)
            last_api_msg = "Backend API (file upload) succeeded."
        else:
            last_api_msg = str(resp)
            ok_local, resp_local = local_predict(df_input)
            if ok_local:
                predictions_df = pd.DataFrame(resp_local["predictions"])
                last_api_msg += " Local inference succeeded after file upload failure."
            else:
                last_api_msg += " Local inference failed: " + str(resp_local)

    st.write(last_api_msg)

# If predictions exist, parse and enrich them
if predictions_df is not None:
    # some backends may return numpy types; convert
    for c in predictions_df.columns:
        if predictions_df[c].dtype == "object":
            # attempt to convert stringified numbers back
            try:
                predictions_df[c] = pd.to_numeric(predictions_df[c], errors="ignore")
            except Exception:
                pass

    # If the API returned the original input columns + scores, OK.
    # Otherwise, join original df_input + predictions_df by index
    if set(["iso_score","ae_score","svm_score","ensemble_score","risky"]).issubset(predictions_df.columns):
        df_result = predictions_df.copy()
    else:
        # join
        df_result = pd.concat([df_input.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)

    # Ensure numeric columns exist
    for col in ["ensemble_score","iso_score","ae_score","svm_score","risky"]:
        if col not in df_result.columns:
            df_result[col] = 0

    # allow local threshold override
    df_result["risky_local"] = (df_result["ensemble_score"] > manual_threshold).astype(int)

    # parse timestamp
    df_result = parse_timestamp_column(df_result)

    # ---------- Main layout with tabs ----------
    st.markdown("## Results")
    st.markdown(f"**Rows:** {len(df_result)} — **Risky (backend):** {int(df_result['risky'].sum())} — **Risky (local threshold):** {int(df_result['risky_local'].sum())}")

    tabs = st.tabs(["Overview", "Table", "Charts", "Explain"])

    # ---------- Overview ----------
    with tabs[0]:
        st.subheader("Quick Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total logs", len(df_result))
        c2.metric("Backend-risky", int(df_result["risky"].sum()))
        c3.metric("Local-risky", int(df_result["risky_local"].sum()))
        avg_score = float(df_result["ensemble_score"].mean())
        c4.metric("Avg ensemble score", f"{avg_score:.3f}")

        st.markdown("### Top tactics (risky logs)")
        risky_tactics = df_result[df_result["risky"]==1]["rule.mitre.tactic"].value_counts().rename_axis("tactic").reset_index(name="count")
        if not risky_tactics.empty:
            fig = px.bar(risky_tactics, x="tactic", y="count", title="Top tactics among backend-flagged risky logs")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risky tactics to show")

        st.markdown("### Top agents by risky count")
        top_agents = df_result[df_result["risky"]==1]["agent.name"].value_counts().head(10).rename_axis("agent").reset_index(name="count")
        if not top_agents.empty:
            fig2 = px.bar(top_agents, x="agent", y="count", title="Top agents generating risky logs")
            st.plotly_chart(fig2, use_container_width=True)

    # ---------- Table ----------
    with tabs[1]:
        st.subheader("Predictions table")
        show_only_risky = st.checkbox("Show only backend-risky rows", value=False)
        df_show = df_result[df_result["risky"]==1] if show_only_risky else df_result
        st.dataframe(df_show, height=500)

        # Download risky logs CSV
        risky_df = df_result[df_result["risky"]==1]
        if not risky_df.empty:
            csv_bytes = risky_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download risky logs (CSV)", csv_bytes, file_name="risky_logs.csv")
        else:
            st.info("No risky logs to download")

    # ---------- Charts ----------
    with tabs[2]:
        st.subheader("Score distributions")
        try:
            fig_hist = px.histogram(df_result, x="ensemble_score", nbins=50, title="Ensemble Score Distribution")
            fig_hist.add_vline(x=manual_threshold, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception:
            st.write("Falling back to Matplotlib histogram")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.hist(df_result["ensemble_score"].dropna(), bins=50)
            ax.axvline(manual_threshold, color="red", linestyle="--")
            st.pyplot(fig)


    # ---------- Explain ----------
    with tabs[3]:
        st.subheader("Explainability (basic)")
        st.markdown("This section provides quick, interpretable signals to help triage why logs were flagged.")
        # Model contribution (which model produced the highest normalized score per row)
        # We compare iso_score, ae_score, svm_score after normalization (they should already be normalized 0-1)
        df_result["top_model"] = df_result[["iso_score","ae_score","svm_score"]].idxmax(axis=1)
        top_model_counts = df_result["top_model"].value_counts().rename_axis("model").reset_index(name="count")
        st.write("Top scoring model counts (which model 'won' per-row):")
        st.table(top_model_counts)

        st.markdown("### Risky tactics breakdown")
        tactics_ct = df_result[df_result["risky"]==1]["rule.mitre.tactic"].value_counts().reset_index()
        tactics_ct.columns = ["tactic", "count"]
        st.bar_chart(tactics_ct.set_index("tactic"))

        st.markdown("### Example risky logs (sample)")
        if not risky_df.empty:
            st.dataframe(risky_df.head(10))
        else:
            st.info("No risky examples to show")

    # End result UI
    st.success("Done — use tabs above to explore results")


# Footer / info
st.markdown("---")
st.caption("Tip: If the API returns errors (or JSON NaN issues), enable 'Use backend API' = False to run inference locally (requires running Streamlit in same environment as backend code).")
