"""
app.py  -  Song Popularity Predictor (combined backend + frontend)
============================================================
Starts the Flask REST API in a background thread, then runs the
Streamlit UI in the main thread.

Usage:
    streamlit run app.py

The Flask API will be available at http://localhost:5000
The Streamlit UI will be available at http://localhost:8501

Requirements:
    pip install numpy pandas scikit-learn flask flask-cors streamlit plotly requests statsmodels
"""

import json
import os
import pickle
import threading
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Paths (all relative to this single file) ─────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "data", "song_data.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FLASK BACKEND
# ══════════════════════════════════════════════════════════════════════════════
flask_app = Flask(__name__)
CORS(flask_app)

FEATURES = [
    "song_duration_ms", "acousticness", "danceability", "energy",
    "instrumentalness", "key", "liveness", "loudness", "audio_mode",
    "speechiness", "tempo", "time_signature", "audio_valence",
]

def _load_artefacts():
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"),      "rb") as f: model   = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),        "rb") as f: scaler  = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f: feats   = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json"))          as f: metrics = json.load(f)
    return model, scaler, feats, metrics

try:
    MODEL, SCALER, _FEATURES, METRICS = _load_artefacts()
except Exception as exc:
    MODEL = SCALER = _FEATURES = METRICS = None

def _build_input(data):
    return np.array([[float(data.get(f, 0)) for f in FEATURES]])

def _label(score):
    if score >= 80: return "Very High"
    if score >= 65: return "High"
    if score >= 50: return "Moderate"
    if score >= 35: return "Low"
    return "Very Low"

@flask_app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})

@flask_app.route("/metrics")
def api_metrics():
    if METRICS is None: return jsonify({"error": "Model not loaded"}), 503
    return jsonify(METRICS)

@flask_app.route("/feature_importance")
def feature_importance():
    if METRICS is None: return jsonify({"error": "Model not loaded"}), 503
    ranked = sorted(METRICS.get("feature_importances", {}).items(), key=lambda x: x[1], reverse=True)
    return jsonify([{"feature": k, "importance": v} for k, v in ranked])

@flask_app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None: return jsonify({"error": "Model not loaded"}), 503
    data = request.get_json(force=True)
    if not data: return jsonify({"error": "No JSON body"}), 400
    try:
        score = float(MODEL.predict(SCALER.transform(_build_input(data)))[0])
        score = round(max(0.0, min(100.0, score)), 2)
        return jsonify({"predicted_popularity": score, "label": _label(score),
                        "input_features": {f: float(data.get(f, 0)) for f in FEATURES}})
    except Exception as e:
        return jsonify({"error": str(e)}), 422

@flask_app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if MODEL is None: return jsonify({"error": "Model not loaded"}), 503
    payload = request.get_json(force=True)
    if not isinstance(payload, list): return jsonify({"error": "Expected JSON array"}), 400
    try:
        rows    = [_build_input(item)[0] for item in payload]
        scores  = MODEL.predict(SCALER.transform(np.array(rows)))
        results = []
        for item, sc in zip(payload, scores):
            s = round(float(max(0.0, min(100.0, sc))), 2)
            results.append({"song_name": item.get("song_name", "Unknown"),
                            "predicted_popularity": s, "label": _label(s)})
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

def _run_flask():
    flask_app.run(port=5000, debug=False, use_reloader=False)

# Start Flask in a daemon thread (dies when Streamlit exits)
_flask_thread = threading.Thread(target=_run_flask, daemon=True)
_flask_thread.start()

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT FRONTEND
# ══════════════════════════════════════════════════════════════════════════════
API_BASE = "http://localhost:5000"

st.set_page_config(
    page_title="Song Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def api_predict(payload):
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "API not reachable — using local model."
    except Exception as e:
        return None, str(e)

def predict_local(payload):
    try:
        with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f: model  = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"),   "rb") as f: scaler = pickle.load(f)
        row   = np.array([[float(payload.get(feat, 0)) for feat in FEATURES]])
        score = round(float(max(0.0, min(100.0, model.predict(scaler.transform(row))[0]))), 2)
        labels = [(80,"Very High"),(65,"High"),(50,"Moderate"),(35,"Low"),(0,"Very Low")]
        label  = next(l for thresh, l in labels if score >= thresh)
        return {"predicted_popularity": score, "label": label}, None
    except Exception as e:
        return None, f"Local prediction failed: {e}"

def popularity_gauge(score, label):
    color = "#22c55e" if score >= 80 else "#3b82f6" if score >= 65 else "#f59e0b" if score >= 50 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": " / 100", "font": {"size": 36}},
        title={"text": f"Predicted Popularity<br><span style='font-size:18px'>{label}</span>", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "white", "borderwidth": 2, "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0,  35], "color": "#fee2e2"},
                {"range": [35, 50], "color": "#fef3c7"},
                {"range": [50, 65], "color": "#dbeafe"},
                {"range": [65, 80], "color": "#dcfce7"},
                {"range": [80,100], "color": "#bbf7d0"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "value": score},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  .metric-card {
    background: #f7f8fa; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 16px 20px; text-align: center;
  }
  .metric-card h2 { margin: 0; font-size: 2rem; color: #3b82d4; }
  .metric-card p  { margin: 4px 0 0; font-size: 0.85rem; color: #57606a; }
  .section-title  { font-size: 1.2rem; font-weight: 600; margin-bottom: 8px; color: #1f2328; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎵 Song Popularity")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Predict Popularity", "Dataset Explorer", "Model Performance", "Feature Analysis"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Powered by Random Forest · Flask · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "Predict Popularity":
    st.title("Song Popularity Predictor")
    st.markdown("Adjust the audio features below and click **Predict** to see the predicted popularity score.")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-title">Audio Features</div>', unsafe_allow_html=True)
        inputs = {}
        with st.expander("Rhythm & Energy", expanded=True):
            inputs["danceability"]   = st.slider("Danceability",   0.0, 1.0,   0.5,   0.01)
            inputs["energy"]         = st.slider("Energy",         0.0, 1.0,   0.7,   0.01)
            inputs["tempo"]          = st.slider("Tempo (BPM)",   40.0, 220.0, 120.0, 0.5)
            inputs["time_signature"] = st.select_slider("Time Signature", options=[1,2,3,4,5,6,7], value=4)
        with st.expander("Tone & Mood", expanded=True):
            inputs["acousticness"]  = st.slider("Acousticness",   0.0, 1.0,  0.1, 0.01)
            inputs["audio_valence"] = st.slider("Valence (Mood)", 0.0, 1.0,  0.5, 0.01)
            inputs["loudness"]      = st.slider("Loudness (dB)", -40.0, 0.0, -5.0, 0.1)
            inputs["audio_mode"]    = st.radio("Mode", options=[0, 1],
                                               format_func=lambda x: "Minor" if x == 0 else "Major",
                                               horizontal=True)
        with st.expander("Other Features", expanded=False):
            inputs["instrumentalness"] = st.slider("Instrumentalness", 0.0, 1.0,  0.0,   0.001)
            inputs["liveness"]         = st.slider("Liveness",         0.0, 1.0,  0.15,  0.01)
            inputs["speechiness"]      = st.slider("Speechiness",      0.0, 1.0,  0.05,  0.01)
            inputs["key"]              = st.select_slider("Key", options=list(range(12)), value=5,
                                             format_func=lambda x: ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][x])
            inputs["song_duration_ms"] = st.number_input("Duration (ms)", 30000, 600000, 210000, 1000)

    with col_r:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        predict_btn = st.button("Predict Popularity", type="primary", use_container_width=True)

        if predict_btn or "last_result" in st.session_state:
            if predict_btn:
                result, err = api_predict(inputs)
                if err:
                    result, err = predict_local(inputs)
                if err:
                    st.error(err)
                    st.info("Tip: Run `python backend/train_model.py` first to train the model.")
                else:
                    st.session_state["last_result"] = result

            if "last_result" in st.session_state:
                r     = st.session_state["last_result"]
                score = r["predicted_popularity"]
                label = r["label"]

                st.plotly_chart(popularity_gauge(score, label), use_container_width=True)
                st.markdown(f"""
                <div class="metric-card">
                  <h2>{score}</h2>
                  <p>Popularity Score out of 100 &nbsp;|&nbsp; {label}</p>
                </div>""", unsafe_allow_html=True)
                st.markdown("---")
                if score >= 80:   st.success("This track has a **very high** chance of being popular!")
                elif score >= 65: st.info("**Above average** popularity predicted.")
                elif score >= 50: st.warning("**Moderate** popularity. Consider tweaking energy or valence.")
                elif score >= 35: st.warning("**Below average** predicted. Significant adjustments may be needed.")
                else:             st.error("**Low popularity** predicted. Major changes recommended.")

                st.markdown("**Your Song's Audio Profile**")
                norm_features = ["danceability","energy","acousticness","audio_valence","liveness","speechiness"]
                norm_values   = [float(inputs.get(f, 0)) for f in norm_features]
                radar_fig = go.Figure(go.Scatterpolar(
                    r=norm_values + [norm_values[0]], theta=norm_features + [norm_features[0]],
                    fill='toself', line_color='#3b82d4', fillcolor='rgba(59,130,212,0.2)',
                ))
                radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                        showlegend=False, height=300,
                                        margin=dict(l=30, r=30, t=30, b=30))
                st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Adjust the sliders on the left and click **Predict Popularity**.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    df = load_dataset()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h2>{len(df):,}</h2><p>Total Songs</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h2>{df["song_popularity"].mean():.1f}</h2><p>Avg Popularity</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h2>{df["song_popularity"].max()}</h2><p>Max Popularity</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h2>{df["song_popularity"].min()}</h2><p>Min Popularity</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Distribution", "Correlations", "Raw Data"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="song_popularity", nbins=40, title="Popularity Distribution",
                               color_discrete_sequence=["#3b82d4"])
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            feat_sel = st.selectbox("Feature vs Popularity", [f for f in FEATURES if f != "song_duration_ms"])
            fig2 = px.scatter(df.sample(min(2000, len(df)), random_state=42),
                              x=feat_sel, y="song_popularity", opacity=0.4, trendline="ols",
                              color_discrete_sequence=["#7c5cd8"], title=f"{feat_sel} vs Popularity")
            st.plotly_chart(fig2, use_container_width=True)
        fig3 = px.box(df, x="audio_mode", y="song_popularity",
                      labels={"audio_mode": "Mode (0=Minor, 1=Major)"}, title="Popularity by Audio Mode",
                      color="audio_mode", color_discrete_sequence=["#ef4444","#22c55e"])
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        corr = df[FEATURES + ["song_popularity"]].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                             title="Feature Correlation Matrix", color_continuous_scale="RdBu_r")
        fig_corr.update_layout(height=550)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.markdown(f"Showing top **{min(500, len(df))}** rows")
        pop_filter = st.slider("Filter by minimum popularity", 0, 100, 0)
        filtered = df[df["song_popularity"] >= pop_filter]
        st.dataframe(filtered.head(500), use_container_width=True, height=400)
        st.download_button("Download Filtered CSV", filtered.to_csv(index=False),
                           file_name="filtered_songs.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")
    metrics = load_metrics()

    if metrics is None:
        st.warning("Model metrics not found. Please run `python backend/train_model.py` first.")
    else:
        st.markdown(f"**Best Model:** `{metrics['best_model']}`")
        tm = metrics["test_metrics"]
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><h2>{tm["MAE"]}</h2><p>Mean Absolute Error</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><h2>{tm["RMSE"]}</h2><p>Root Mean Square Error</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><h2>{tm["R2"]}</h2><p>R² Score</p></div>', unsafe_allow_html=True)
        st.markdown("---")

        all_m = metrics.get("all_models", {})
        if all_m:
            df_m = pd.DataFrame([{"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]}
                                  for k, v in all_m.items()])
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.bar(df_m, x="Model", y="R²", title="R² Score by Model",
                                       color="R²", color_continuous_scale="Blues", text_auto=".3f"),
                                use_container_width=True)
            with col2:
                st.plotly_chart(px.bar(df_m, x="Model", y="RMSE", title="RMSE by Model",
                                       color="RMSE", color_continuous_scale="Reds_r", text_auto=".3f"),
                                use_container_width=True)
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;padding:20px">
          <b>Training rows:</b> {metrics.get('train_rows','N/A'):,} &nbsp;|&nbsp;
          <b>Test rows:</b> {metrics.get('test_rows','N/A'):,}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Analysis":
    st.title("Feature Analysis")
    metrics = load_metrics()
    df = load_dataset()

    if metrics and metrics.get("feature_importances"):
        fi    = metrics["feature_importances"]
        df_fi = pd.DataFrame(sorted(fi.items(), key=lambda x: x[1], reverse=True),
                             columns=["Feature", "Importance"])
        col1, col2 = st.columns([1, 1])
        with col1:
            fig_fi = px.bar(df_fi, x="Importance", y="Feature", orientation="h",
                            title="Feature Importances (from best model)",
                            color="Importance", color_continuous_scale="Blues", text_auto=".4f")
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig_fi, use_container_width=True)
        with col2:
            top_feat = df_fi["Feature"].iloc[0]
            st.markdown(f"**Top feature: `{top_feat}`**")
            fig_top = px.scatter(df.sample(min(3000, len(df)), random_state=1),
                                 x=top_feat, y="song_popularity", opacity=0.4, trendline="ols",
                                 title=f"Top Feature: {top_feat} vs Popularity",
                                 color_discrete_sequence=["#3b82d4"])
            st.plotly_chart(fig_top, use_container_width=True)

        top4 = df_fi["Feature"].head(4).tolist()
        st.markdown("**Top-4 Features vs Popularity**")
        fig_multi = make_subplots(rows=2, cols=2,
                                  subplot_titles=[f"{f} vs Popularity" for f in top4])
        sample = df.sample(min(1500, len(df)), random_state=42)
        for idx, feat in enumerate(top4):
            row, col = divmod(idx, 2)
            fig_multi.add_trace(
                go.Scatter(x=sample[feat], y=sample["song_popularity"], mode="markers", opacity=0.4,
                           marker=dict(color=["#3b82d4","#7c5cd8","#22c55e","#f59e0b"][idx], size=4),
                           name=feat),
                row=row+1, col=col+1)
        fig_multi.update_layout(height=550, showlegend=False)
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.warning("Feature importances not found. Please run `python backend/train_model.py` first.")
