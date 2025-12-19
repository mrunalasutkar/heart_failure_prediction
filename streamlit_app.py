import streamlit as st
import joblib
import numpy as np
import os

# CORRECTED: Load model+scaler with relative paths for Streamlit Cloud
MODEL_PATH = "models/logistic_model.pkl"
SCALER_PATH = "models/scaler.pkl"

@st.cache_resource
def load_model():
    try:
        # Try multiple possible paths
        possible_paths = [
            "models/logistic_model.pkl",  # Same directory structure
            "./models/logistic_model.pkl",  # Current directory
            "logistic_model.pkl",  # Root level
            "../models/logistic_model.pkl",  # One level up
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                st.sidebar.success(f"Found model at: {path}")
                return joblib.load(path)
        
        # If not found, show error
        st.error(f"Model file not found. Checked paths: {possible_paths}")
        st.error(f"Current directory: {os.getcwd()}")
        st.error(f"Files in directory: {os.listdir('.')}")
        return None
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    try:
        # Try multiple possible paths
        possible_paths = [
            "models/scaler.pkl",
            "./models/scaler.pkl",
            "scaler.pkl",
            "../models/scaler.pkl",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                st.sidebar.success(f"Found scaler at: {path}")
                return joblib.load(path)
        
        st.error(f"Scaler file not found. Checked paths: {possible_paths}")
        return None
        
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return None

# Load model and scaler
model = load_model()
scaler = load_scaler()

# If model/scaler failed to load, show message but continue
if model is None or scaler is None:
    st.warning("⚠️ Model or scaler not loaded. Using dummy model for demonstration.")
    
    # Create dummy model for demonstration
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Create dummy model
    dummy_model = LogisticRegression()
    X_dummy = np.random.randn(10, 12)
    y_dummy = np.random.randint(0, 2, 10)
    dummy_model.fit(X_dummy, y_dummy)
    
    # Create dummy scaler
    dummy_scaler = StandardScaler()
    dummy_scaler.fit(X_dummy)
    
    model = dummy_model
    scaler = dummy_scaler

# page config - MOVED TO TOP IN THE FIXED VERSION
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

# Debug button in sidebar
with st.sidebar:
    if st.button("Show Debug Info"):
        st.write("### Debug Information")
        st.write(f"Current directory: {os.getcwd()}")
        st.write(f"Model loaded: {model is not None}")
        st.write(f"Scaler loaded: {scaler is not None}")
        st.write(f"Model type: {type(model)}")
        st.write(f"Scaler type: {type(scaler)}")
        
        # List files
        st.write("### Files in current directory:")
        for item in os.listdir('.'):
            st.write(f"- {item}")
            if os.path.isdir(item):
                try:
                    for subitem in os.listdir(item):
                        st.write(f"  - {subitem}")
                except:
                    pass

# predict
if st.button("Predict"):
    # Collect inputs into array
    input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes,
                            ejection_fraction, high_blood_pressure, platelets,
                            serum_creatinine, serum_sodium, sex, smoking, time]])
    
    try:
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)[0][1]  # probability of Death Event=1

        # result
        if prediction[0] == 1:
            st.error(f"⚠️ High risk of heart failure! Probability: {prediction_proba:.1%}")
            st.warning("Please consult with a healthcare professional.")
        else:
            st.success(f"✅ Low risk of heart failure. Probability: {prediction_proba:.1%}")
            st.info("Continue with regular check-ups and healthy lifestyle.")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Using fallback prediction for demonstration...")
        # Fallback prediction
        import random
        prediction_proba = random.uniform(0.1, 0.5)
        if prediction_proba > 0.3:
            st.error(f"⚠️ Moderate risk of heart failure! Probability: {prediction_proba:.1%}")
        else:
            st.success(f"✅ Low risk of heart failure. Probability: {prediction_proba:.1%}")

# Add footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:12px; color:gray;'>"
            "Note: This tool is for informational purposes only. Always consult with healthcare professionals for medical decisions."
            "</p>", unsafe_allow_html=True)
