import streamlit as st
import pickle
import joblib
import numpy as np

# load model+scaler
MODEL_PATH = r"D:\heart_failure_prediction\models\logistic_model.pkl"
SCALER_PATH = r"D:\heart_failure_prediction\models\scaler.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()

# page config
st.set_page_config(
    page_title="Heart Failure Predictor",
    page_icon="🫀",
    layout="centered"
)

# page title
st.markdown("<h1 style='text-align:center;color:#FF4B4B;'>❤️‍🩹Heart Failure Prediction</h1>",unsafe_allow_html=True)
# subtitle
st.markdown("<p style='text-align:center; color:#FFD700;'>Predict Your Heart Risk Accurately</p>",unsafe_allow_html=True)

# divider
st.markdown("<hr style='border:none; height:2px; background-color: #FF4B4B; margin:20px 0;'>",unsafe_allow_html=True)

st.markdown("<h5 style='text-align:left;color:#89CFF0;'>👩🏻Patient Details</h5>",unsafe_allow_html=True)

# input features
col6, col7, col8 = st.columns(3)
with col6:
    anaemia = st.radio("Anaemia", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
with col7:
    diabetes = st.radio("Diabetes", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
with col8:
    high_blood_pressure = st.radio("High Blood Pressure", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")

col9, col10, col11 = st.columns(3)
with col9:
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
with col10:
    smoking = st.radio("Smoking", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")


col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
with col2:
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0, value=0)
with col3:
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=50)


col4, col5, col6 = st.columns(3)
with col4:
    platelets = st.number_input("Platelets (k/mL)", min_value=0.0, value=250000.0, format="%.1f")
with col5:
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.0, format="%.2f")
with col6:
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=160, value=135)

time = st.number_input("Follow-up time (days)", min_value=0, value=100)

# divider
st.markdown("<hr style='border:none; height:2px; background-color: #FF4B4B; margin:20px 0;'>",unsafe_allow_html=True)

# predict
if st.button("Predict"):
    # Collect inputs into array
    input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes,
                            ejection_fraction, high_blood_pressure, platelets,
                            serum_creatinine, serum_sodium, sex, smoking, time]])
    
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0][1]  # probability of Death Event=1

    # result
    if prediction[0] == 1:
        st.error(f"High risk of heart failure! Probability: {prediction_proba:.2f}")
    else:
        st.success(f"Low risk of heart failure. Probability: {prediction_proba:.2f}")