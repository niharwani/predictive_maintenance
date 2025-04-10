import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import os

st.set_page_config(page_title='Predictive Maintenance Dashboard', layout='wide')

st.title("🛠️ Predictive Maintenance Dashboard")
st.markdown("Upload your processed dataset (or use the built-in sample below).")

# 📁 File upload or fallback to sample
uploaded_file = st.file_uploader("Upload Processed Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Uploaded dataset loaded!")
else:
    st.info("ℹ️ No file uploaded. Using default sample dataset.")
    df = pd.read_csv("data/sample_processed_FD001.csv")

    # 📥 Download sample option
    with open("data/sample_processed_FD001.csv", "rb") as file:
        st.download_button(
            label="📥 Download Sample Dataset",
            data=file,
            file_name="sample_processed_FD001.csv",
            mime="text/csv"
        )

# 🧠 Select engine
engine_ids = df['unit_number'].unique()
selected_engine = st.selectbox("Select Engine", engine_ids)

# 📊 Show sensor trends
engine_data = df[df['unit_number'] == selected_engine]

st.subheader(f"📈 Sensor Readings - Engine {selected_engine}")
st.line_chart(engine_data.set_index('time_in_cycles')[['sensor_2', 'sensor_3', 'sensor_4']])

# 🔍 Predict RUL
model_path = 'models/xgb_model.pkl'

if os.path.exists(model_path):
    import joblib
    model = joblib.load(model_path)

    st.subheader("🔮 Predicted Remaining Useful Life (RUL)")
    features = [col for col in df.columns if 'sensor' in col]
    X_latest = engine_data[features].iloc[-1:].values
    predicted_rul = model.predict(X_latest)[0]
    st.metric(label="Predicted RUL", value=f"{predicted_rul:.2f} cycles")

    # Health status
    if predicted_rul < 20:
        st.error("⚠️ Immediate Maintenance Required!")
    elif predicted_rul < 50:
        st.warning("🔧 Maintenance Recommended Soon")
    else:
        st.success("✅ Engine Operating Normally")

else:
    st.error("❌ Model file not found. Please upload or add it to 'models/xgb_model.pkl'")
