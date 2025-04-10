# 🛠️ Predictive Maintenance Using Machine Learning

This project leverages machine learning to predict the **Remaining Useful Life (RUL)** of industrial machines using historical sensor data from the NASA CMAPSS dataset. It features a fully interactive web dashboard built with **Streamlit** and is deployed live for real-time use.

---

## 🚀 Live Demo
🔗 [Launch App on Streamlit](https://predictivemaintenancebyniharwani.streamlit.app/)

---

## 📌 Features
- Upload either **processed CSV** or **raw `train_FD001.txt`**
- Smart in-app preprocessing of raw NASA data (no code needed)
- Real-time RUL predictions using trained **XGBoost model**
- Sensor data visualization (sensor_2, sensor_3, sensor_4)
- Automatic health status alerts:
  - ✅ Healthy
  - ⚠️ Maintenance Recommended
  - 🔧 Critical - Immediate Action Required
- Downloadable sample dataset for quick testing

---

## 📊 Dataset
- **NASA CMAPSS** (FD001 subset)
- Simulated turbofan engine degradation dataset
- Each engine has sensor readings over time until failure
- RUL is calculated as `max_cycle - current_cycle`

[Kaggle Dataset Link](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## 🧠 Models Used
- ✅ **XGBoost Regressor** (trained on normalized sensor data)
- 🔄 Future Enhancement: LSTM (deep learning for sequence modeling)

---

## 📂 How to Run Locally
```bash
# Clone the repo
https://github.com/niharwani/predictive_maintenance.git
cd predictive_maintenance

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py
```

---

## 📸 Screenshot
![Screenshot](https://github.com/user-attachments/assets/e2411a3c-15ce-46b2-a2b4-48240a54d608)


---

## 📌 File Structure
```
├── app/                  # Streamlit frontend
│   └── app.py
├── data/                 # Raw and sample data files
│   └── sample_processed_FD001.csv
├── models/               # Trained model
│   └── xgb_model.pkl
├── notebooks/            # Preprocessing & modeling
├── utils/                # Helper scripts (future)
├── requirements.txt
├── README.md
```

---

## 🙌 Credits
- Project by **Nihar Wani**
- Built using **Python, Streamlit, XGBoost, pandas, matplotlib**
- Inspired by real-world predictive maintenance in Industry 4.0

---

## 🔗 Connect
- [LinkedIn](https://www.linkedin.com/in/niharwani)
- GitHub: [niharwani](https://github.com/niharwani)

---
