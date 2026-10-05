# 🎵 Song Popularity Prediction

A full-stack machine-learning application that predicts the **popularity score (0-100)** of a song based on its audio features, trained on ~18,000 real songs.
dataset: https://www.kaggle.com/datasets/yasserh/song-popularity-dataset

## 📁 Project Structure

```
song_popularity_app/
├── data/
│   └── song_data.csv          # ~18k song records
├── model/                     # Generated after training
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── metrics.json
├── backend/
│   ├── train_model.py         # Data preprocessing + model training
│   └── app.py                 # Flask REST API (port 5000)
├── frontend/
│   └── app.py                 # Streamlit UI (port 8501)
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python backend/train_model.py
```
This compares **RandomForest**, **GradientBoosting**, and **Ridge** regression and saves the best model.

### 3. Start the Flask API (Terminal 1)
```bash
python backend/app.py
```
API runs at `http://localhost:5000`

### 4. Launch the Streamlit UI (Terminal 2)
```bash
streamlit run frontend/app.py
```
UI opens at `http://localhost:8501`

> **Note:** The frontend can also run in *offline mode* — it falls back to direct model loading if the Flask API is unreachable.

## 🧠 Features Used

| Feature | Description |
|---|---|
| song_duration_ms | Track length in milliseconds |
| danceability | How suitable for dancing (0–1) |
| energy | Perceptual measure of intensity (0–1) |
| acousticness | Confidence of acoustic quality (0–1) |
| instrumentalness | Predicts no vocal content (0–1) |
| loudness | Overall loudness in dB |
| tempo | Estimated tempo in BPM |
| audio_valence | Musical positiveness/mood (0–1) |
| liveness | Presence of audience (0–1) |
| speechiness | Presence of spoken words (0–1) |
| key | Musical key (0–11) |
| audio_mode | Major (1) or minor (0) |
| time_signature | Estimated meter |

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/metrics` | Model metrics + feature importances |
| GET | `/feature_importance` | Ranked feature list |
| POST | `/predict` | Predict popularity for one song |
| POST | `/predict_batch` | Predict for a list of songs |

### Example POST `/predict`
```json
{
  "danceability": 0.7,
  "energy": 0.85,
  "loudness": -4.5,
  "tempo": 128.0,
  "audio_valence": 0.6,
  "acousticness": 0.02,
  "instrumentalness": 0.0,
  "liveness": 0.1,
  "speechiness": 0.05,
  "key": 5,
  "audio_mode": 1,
  "time_signature": 4,
  "song_duration_ms": 210000
}
```

## 📊 UI Pages

1. **🔮 Predict Popularity** – Interactive sliders → gauge chart + radar plot
2. **📊 Dataset Explorer** – Distribution, scatter plots, correlation matrix, raw data
3. **🏆 Model Performance** – MAE / RMSE / R² comparison across all trained models
4. **📈 Feature Analysis** – Feature importance bar chart + scatter grids
