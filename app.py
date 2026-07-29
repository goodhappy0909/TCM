%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
import os # Import os module for path checking

# Constants from the notebook
SEQUENCE_LENGTH = 96  # 24 hours of 15-minute intervals
SENSOR_FEATURES = [
    "usage_kwh", "reactive_kvarh", "power_factor_pct",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekday",
]
DATA_FREQUENCY_MINUTES = 15

st.write("Streamlit 앱 시작...") # Added log

# Load model and scalers (cached for efficiency)
@st.cache_resource
def load_model_and_scalers():
    st.write("모델과 스케일러 로드 시작...") # Added log
    # Use relative paths for deployment
    model_dir = "lesson07_deep_timeseries_outputs"
    model_path = os.path.join(model_dir, "lesson07_tcn_gru_energy_model.keras")
    scaler_path = os.path.join(model_dir, "lesson07_tcn_gru_scalers.joblib")

    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"오류: 모델 파일이 존재하지 않습니다: {model_path}")
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
    if not os.path.exists(scaler_path):
        st.error(f"오류: 스케일러 파일이 존재하지 않습니다: {scaler_path}")
        raise FileNotFoundError(f"스케일러 파일이 존재하지 않습니다: {scaler_path}")

    try:
        model = tf.keras.models.load_model(model_path)
        scalers = joblib.load(scaler_path)
        st.write("모델과 스케일러 로드 성공!") # Added log
        return model, scalers["feature_scaler"], scalers["target_scaler"]
    except Exception as e:
        st.error(f"오류: 모델 또는 스케일러 로드 중 문제 발생: {e}")
        raise e

try:
    model, feature_scaler, target_scaler = load_model_and_scalers()
except FileNotFoundError:
    st.stop() # Stop the app if essential files are missing
except Exception as e:
    st.error(f"초기화 중 치명적인 오류 발생: {e}")
    st.stop()

# Helper function to generate time features for a given timestamp
def get_time_features(timestamp_series):
    hour = timestamp_series.dt.hour + timestamp_series.dt.minute / 60
    dow = timestamp_series.dt.dayofweek
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekday": (dow < 5).astype(float),
    })

# Prediction function using the loaded model and scalers
def predict_next_usage(
    current_sensor_sequence,
    model,
    feature_scaler,
    target_scaler
):
    scaled_input = feature_scaler.transform(
        current_sensor_sequence.reshape(-1, len(SENSOR_FEATURES))
    ).reshape(1, SEQUENCE_LENGTH, len(SENSOR_FEATURES))
    prediction = model.predict(scaled_input, verbose=0).ravel()[0]
    prediction = target_scaler.inverse_transform([[prediction]])[0][0]
    return np.maximum(prediction, 0)

# Streamlit app layout
st.set_page_config(layout="wide", page_title="전력 수요 예측 대시보드")
st.title("미래 전력 수요 예측 대시보드 (TCN-GRU 모델)")

st.markdown("이 대시보드는 UCI 철강 산업 에너지 소비 데이터를 학습한 TCN-GRU 모델을 사용하여 특정 시점부터 24시간 동안의 전력 수요를 예측합니다.")

# User input for the starting prediction time
st.sidebar.header("예측 시작 시간 설정")
prediction_date = st.sidebar.date_input("날짜 선택", datetime.now().date())
prediction_time = st.sidebar.time_input("시간 선택", datetime.now().time())

start_prediction_timestamp = datetime.combine(prediction_date, prediction_time)
st.sidebar.write(f"선택된 예측 시작 시간: {start_prediction_timestamp.strftime('%Y-%m-%d %H:%M')}")

if st.sidebar.button("예측 실행"):
    st.subheader("향후 24시간 전력 수요 예측")

    dummy_base_usage = 50.0 # Example average usage
    dummy_base_reactive = 30.0 # Example average reactive power
    dummy_base_power_factor = 85.0 # Example average power factor

    input_timestamps = pd.to_datetime([
        start_prediction_timestamp - timedelta(minutes=DATA_FREQUENCY_MINUTES * (SEQUENCE_LENGTH - 1 - i))
        for i in range(SEQUENCE_LENGTH)
    ])

    input_time_features_df = get_time_features(input_timestamps)

    current_sequence = np.zeros((SEQUENCE_LENGTH, len(SENSOR_FEATURES)))
    for i in range(SEQUENCE_LENGTH):
        current_sequence[i, SENSOR_FEATURES.index("usage_kwh")] = dummy_base_usage
        current_sequence[i, SENSOR_FEATURES.index("reactive_kvarh")] = dummy_base_reactive
        current_sequence[i, SENSOR_FEATURES.index("power_factor_pct")] = dummy_base_power_factor

    current_sequence[:, SENSOR_FEATURES.index("hour_sin")] = input_time_features_df["hour_sin"].values
    current_sequence[:, SENSOR_FEATURES.index("hour_cos")] = input_time_features_df["hour_cos"].values
    current_sequence[:, SENSOR_FEATURES.index("dow_sin")] = input_time_features_df["dow_sin"].values
    current_sequence[:, SENSOR_FEATURES.index("dow_cos")] = input_time_features_df["dow_cos"].values
    current_sequence[:, SENSOR_FEATURES.index("is_weekday")] = input_time_features_df["is_weekday"].values

    predicted_usages = []
    prediction_timestamps = []

    st.write("예측 진행 중...")
    progress_bar = st.progress(0)

    for i in range(SEQUENCE_LENGTH): # Predict next 96 steps (24 hours)
        next_prediction_time = start_prediction_timestamp + timedelta(minutes=DATA_FREQUENCY_MINUTES * i)

        predicted_kwh = predict_next_usage(current_sequence, model, feature_scaler, target_scaler)
        predicted_usages.append(predicted_kwh)
        prediction_timestamps.append(next_prediction_time)

        current_sequence = np.roll(current_sequence, -1, axis=0)

        new_last_timestamp = next_prediction_time
        new_time_features_df = get_time_features(pd.to_datetime([new_last_timestamp]))

        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("usage_kwh")] = predicted_kwh
        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("reactive_kvarh")] = dummy_base_reactive
        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("power_factor_pct")] = dummy_base_power_factor

        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("hour_sin")] = new_time_features_df["hour_sin"].values[0]
        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("hour_cos")] = new_time_features_df["hour_cos"].values[0]
        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("dow_sin")] = new_time_features_df["dow_sin"].values[0]
        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("dow_cos")] = new_time_features_df["dow_cos"].values[0]
        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("is_weekday")] = new_time_features_df["is_weekday"].values[0]

        progress_bar.progress((i + 1) / SEQUENCE_LENGTH)

    prediction_df = pd.DataFrame({
        "Timestamp": prediction_timestamps,
        "Predicted Usage (kWh)": predicted_usages
    })

    st.dataframe(prediction_df)
    st.line_chart(prediction_df.set_index("Timestamp"))

    st.success("예측 완료!")
