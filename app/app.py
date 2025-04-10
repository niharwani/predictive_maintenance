
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib

st.set_page_config(page_title='Predictive Maintenance Dashboard', layout='wide')
st.title("🛠️ Predictive Maintenance Dashboard")

st.markdown("### Select input mode:")
input_mode = st.radio("Choose data input type:", ["Upload Processed CSV", "Upload Raw NASA Data (.txt)"])

df = None

if input_mode == "Upload Processed CSV":
    uploaded_file = st.file_uploader("Upload Processed Dataset (.csv)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded preprocessed dataset loaded.")
    else:
        st.info("ℹ️ No file uploaded. Using default sample dataset.")
        df = pd.read_csv("data/sample_processed_FD001.csv")

        with open("data/sample_processed_FD001.csv", "rb") as file:
            st.download_button(
                label="📥 Download Sample Dataset",
                data=file,
                file_name="sample_processed_FD001.csv",
                mime="text/csv"
            )

elif input_mode == "Upload Raw NASA Data (.txt)":
    raw_file = st.file_uploader("Upload Raw NASA FD001 File (.txt)", type="txt")
    if raw_file is not None:
        st.info("📊 Preprocessing raw file...")
        column_names = ['unit_number', 'time_in_cycles'] +                            [f'op_setting_{i}' for i in range(1, 4)] +                            [f'sensor_{i}' for i in range(1, 22)]
        df = pd.read_csv(raw_file, sep='\s+', header=None)
        df.columns = column_names

        # Calculate RUL
        rul_df = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
        rul_df.columns = ['unit_number', 'max_cycle']
        df = df.merge(rul_df, on='unit_number')
        df['RUL'] = df['max_cycle'] - df['time_in_cycles']
        df.drop(columns=['max_cycle'], inplace=True)

        st.success("✅ Raw data successfully preprocessed.")

if df is not None:
    engine_ids = df['unit_number'].unique()
    selected_engine = st.selectbox("Select Engine", engine_ids)

    engine_data = df[df['unit_number'] == selected_engine]

    st.subheader(f"📈 Sensor Readings - Engine {selected_engine}")
    st.line_chart(engine_data.set_index('time_in_cycles')[['sensor_2', 'sensor_3', 'sensor_4']])

    model_path = 'models/xgb_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.subheader("🔮 Predicted Remaining Useful Life (RUL)")

        features = [col for col in df.columns if 'sensor' in col]
        X_latest = engine_data[features].iloc[-1:].values
        predicted_rul = model.predict(X_latest)[0]
        st.metric(label="Predicted RUL", value=f"{predicted_rul:.2f} cycles")

        if predicted_rul < 20:
            st.error("⚠️ Immediate Maintenance Required!")
        elif predicted_rul < 50:
            st.warning("🔧 Maintenance Recommended Soon")
        else:
            st.success("✅ Engine Operating Normally")
    else:
        st.error("❌ Model file not found. Please upload or add it to 'models/xgb_model.pkl'")
