%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta

# Constants from the notebook
SEQUENCE_LENGTH = 96  # 24 hours of 15-minute intervals
SENSOR_FEATURES = [
    "usage_kwh", "reactive_kvarh", "power_factor_pct",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekday",
]
DATA_FREQUENCY_MINUTES = 15

# Load model and scalers (cached for efficiency)
@st.cache_resource
def load_model_and_scalers():
    # Use relative paths for deployment
    model_path = "lesson07_deep_timeseries_outputs/lesson07_tcn_gru_energy_model.keras"
    scaler_path = "lesson07_deep_timeseries_outputs/lesson07_tcn_gru_scalers.joblib"

    model = tf.keras.models.load_model(model_path)
    scalers = joblib.load(scaler_path)
    return model, scalers["feature_scaler"], scalers["target_scaler"]

model, feature_scaler, target_scaler = load_model_and_scalers()

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

    # Simulate the last known sequence (e.g., from X_test[-1])
    # For simplicity, we'll use a fixed 'base' sensor reading for usage, reactive, power factor
    # and overlay the time features corresponding to the prediction window.
    # In a real application, this would come from actual live sensor data.

    # Let's use the average of the first three sensor features from X_test as a base
    # You could also load a specific X_test[-1] if desired for a more direct demo.
    # For this dashboard, we'll generate plausible base values.

    # Dummy initial sensor values (non-time features) based on general dataset characteristics
    # These would ideally be the actual last recorded sensor values from the real system.
    dummy_base_usage = 50.0 # Example average usage
    dummy_base_reactive = 30.0 # Example average reactive power
    dummy_base_power_factor = 85.0 # Example average power factor

    # Prepare an initial sequence to feed the model
    # This sequence needs to represent the 'past 96 intervals' leading up to the first prediction.
    # We will simulate this by taking the `start_prediction_timestamp` as the *target* for the first prediction
    # and generating the 96 prior timestamps for the input sequence.

    # Generate timestamps for the input sequence (96 intervals before start_prediction_timestamp)
    input_timestamps = pd.to_datetime([
        start_prediction_timestamp - timedelta(minutes=DATA_FREQUENCY_MINUTES * (SEQUENCE_LENGTH - 1 - i))
        for i in range(SEQUENCE_LENGTH)
    ])

    # Generate time-based features for the input sequence
    input_time_features_df = get_time_features(input_timestamps)

    # Create the full input sequence array
    # Fill non-time features with a plausible constant/average for demonstration
    current_sequence = np.zeros((SEQUENCE_LENGTH, len(SENSOR_FEATURES)))
    for i in range(SEQUENCE_LENGTH):
        current_sequence[i, SENSOR_FEATURES.index("usage_kwh")] = dummy_base_usage
        current_sequence[i, SENSOR_FEATURES.index("reactive_kvarh")] = dummy_base_reactive
        current_sequence[i, SENSOR_FEATURES.index("power_factor_pct")] = dummy_base_power_factor

    # Overlay the generated time features
    current_sequence[:, SENSOR_FEATURES.index("hour_sin")] = input_time_features_df["hour_sin"].values
    current_sequence[:, SENSOR_FEATURES.index("hour_cos")] = input_time_features_df["hour_cos"].values
    current_sequence[:, SENSOR_FEATURES.index("dow_sin")] = input_time_features_df["dow_sin"].values
    current_sequence[:, SENSOR_FEATURES.index("dow_cos")] = input_time_features_df["dow_cos"].values
    current_sequence[:, SENSOR_FEATURES.index("is_weekday")] = input_time_features_df["is_weekday"].values

    # Store predictions
    predicted_usages = []
    prediction_timestamps = []

    # Predict for the next 24 hours (SEQUENCE_LENGTH steps)
    st.write("예측 진행 중...")
    progress_bar = st.progress(0)

    for i in range(SEQUENCE_LENGTH): # Predict next 96 steps (24 hours)
        next_prediction_time = start_prediction_timestamp + timedelta(minutes=DATA_FREQUENCY_MINUTES * i)

        # Get time features for the *next* prediction point (which will be the last element of the new sequence)
        # This logic is simplified; ideally, current_sequence's time features should be for its specific past times.
        # For recursive prediction, we only need to update the *last* row's time features IF we're using current_sequence.

        # For iterative prediction, the 'current_sequence' is effectively shifted forward.
        # The prediction for `next_prediction_time` is based on the sequence ending at `next_prediction_time - 15min`

        # Make a prediction for the current `current_sequence`
        predicted_kwh = predict_next_usage(current_sequence, model, feature_scaler, target_scaler)
        predicted_usages.append(predicted_kwh)
        prediction_timestamps.append(next_prediction_time)

        # Prepare the next input sequence for the next prediction (recursive step)
        # Shift the sequence: remove the oldest step, add the new prediction at the end.
        # This requires reconstructing the 'newest' sensor features.

        # Newest non-time sensor features are assumed to be the predicted usage for `usage_kwh`,
        # and for reactive_kvarh/power_factor_pct, we'll keep the dummy base for now, or use complex logic.
        # For simplicity, let's assume `usage_kwh` becomes the predicted value, and others remain base.

        # Drop the oldest entry
        current_sequence = np.roll(current_sequence, -1, axis=0)

        # Update the last entry (the newest one) with the new predicted usage and time features
        new_last_timestamp = next_prediction_time # This is the timestamp for the data that was just predicted
        new_time_features_df = get_time_features(pd.to_datetime([new_last_timestamp]))

        # Update the last row of the sequence
        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("usage_kwh")] = predicted_kwh
        # For reactive and power_factor, we'll keep the base values for this demo, or use a more complex forecast for them
        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("reactive_kvarh")] = dummy_base_reactive # Simplified
        current_sequence[SEQUENCE_LENGTH - 1, SENSOR_FEATURES.index("power_factor_pct")] = dummy_base_power_factor # Simplified

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
