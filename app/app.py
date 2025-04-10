
import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import os

st.set_page_config(page_title='Predictive Maintenance Dashboard', layout='wide')

st.title("🛠️ Predictive Maintenance Dashboard")
st.markdown("This app predicts the Remaining Useful Life (RUL) of industrial engines using a trained XGBoost model.")

uploaded_file = st.file_uploader("Upload Processed Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")

    engine_ids = df['unit_number'].unique()
    selected_engine = st.selectbox("Select Engine", engine_ids)

    engine_data = df[df['unit_number'] == selected_engine]

    st.subheader(f"Sensor Readings - Engine {selected_engine}")
    st.line_chart(engine_data.set_index('time_in_cycles')[['sensor_2', 'sensor_3', 'sensor_4']])

    # Load model
    model_path = 'models/xgb_model.pkl'
    if os.path.exists(model_path):
        import joblib
        model = joblib.load(model_path)

        st.subheader("📉 Predicted RUL")
        features = [col for col in df.columns if 'sensor' in col]
        X_latest = engine_data[features].iloc[-1:].values
        predicted_rul = model.predict(X_latest)[0]
        st.metric(label="Predicted Remaining Useful Life (RUL)", value=f"{predicted_rul:.2f} cycles")

        if predicted_rul < 20:
            st.warning("⚠️ Maintenance Required Soon!")
        elif predicted_rul < 50:
            st.info("🔧 Maintenance Advised in Near Future.")
        else:
            st.success("✅ Engine Operating Normally.")
    else:
        st.error("Model file not found. Please add the trained model to 'models/xgb_model.pkl'")
else:
    st.info("Upload a processed dataset to get started.")
