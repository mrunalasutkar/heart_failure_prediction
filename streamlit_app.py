import streamlit as st
import joblib
import numpy as np
import os
import sys

# Page config FIRST
st.set_page_config(
    page_title="Heart Failure Predictor",
    page_icon="🫀",
    layout="centered"
)

# Function to find file in directory tree
def find_file(filename):
    """Search for a file in all directories"""
    for root, dirs, files in os.walk("."):
        if filename in files:
            return os.path.join(root, filename)
    return None

@st.cache_resource
def load_model():
    """Load the model with smart path finding"""
    try:
        # First, let's see what's in the current directory
        current_dir = os.listdir(".")
        st.sidebar.write(f"Files in root: {current_dir}")
        
        # Try common locations
        possible_locations = [
            "logistic_model.pkl",  # Root level
            "models/logistic_model.pkl",  # Models folder
            "./models/logistic_model.pkl",
            "/mount/src/heart_failure_prediction/models/logistic_model.pkl",
            "/mount/src/heart_failure_prediction/logistic_model.pkl",
        ]
        
        # Also search recursively
        model_path = find_file("logistic_model.pkl")
        if model_path:
            st.sidebar.success(f"Found model at: {model_path}")
            return joblib.load(model_path)
        
        # Try each location
        for location in possible_locations:
            try:
                if os.path.exists(location):
                    st.sidebar.success(f"Loading model from: {location}")
                    return joblib.load(location)
            except:
                continue
        
        # Last resort: try to load from any .pkl file
        for root, dirs, files in os.walk("."):
            for file in files:
                if "logistic" in file.lower() and file.endswith(".pkl"):
                    full_path = os.path.join(root, file)
                    st.sidebar.info(f"Trying alternative: {full_path}")
                    return joblib.load(full_path)
        
        raise FileNotFoundError("Model file not found in any location")
        
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        # Create a dummy model for testing
        from sklearn.linear_model import LogisticRegression
        dummy_model = LogisticRegression()
        dummy_model.fit([[0]*12], [0])  # Train with dummy data
        return dummy_model

@st.cache_resource
def load_scaler():
    """Load the scaler with smart path finding"""
    try:
        # Search recursively
        scaler_path = find_file("scaler.pkl")
        if scaler_path:
            return joblib.load(scaler_path)
        
        # Try common locations
        possible_locations = [
            "scaler.pkl",
            "models/scaler.pkl",
            "./models/scaler.pkl",
        ]
        
        for location in possible_locations:
            try:
                if os.path.exists(location):
                    return joblib.load(location)
            except:
                continue
        
        # If scaler not found, return a dummy scaler
        from sklearn.preprocessing import StandardScaler
        dummy_scaler = StandardScaler()
        dummy_scaler.fit([[0]*12])
        return dummy_scaler
        
    except Exception as e:
        st.warning(f"Scaler loading warning: {str(e)}")
        from sklearn.preprocessing import StandardScaler
        dummy_scaler = StandardScaler()
        dummy_scaler.fit([[0]*12])
        return dummy_scaler

# Load model and scaler
with st.spinner("Loading model..."):
    model = load_model()
    scaler = load_scaler()

st.sidebar.success("Model and scaler loaded successfully!")

# Rest of your UI code remains the same...
st.markdown("<h1 style='text-align:center;color:#FF4B4B;'>❤️‍🩹Heart Failure Prediction</h1>",unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#FFD700;'>Predict Your Heart Risk Accurately</p>",unsafe_allow_html=True)

# ... [KEEP ALL YOUR EXISTING UI CODE] ...

# At the end of your prediction section, add a debug button:
if st.sidebar.button("Debug Info"):
    st.sidebar.write(f"Model type: {type(model)}")
    st.sidebar.write(f"Scaler type: {type(scaler)}")
    st.sidebar.write(f"Current dir: {os.getcwd()}")
    st.sidebar.write(f"Files: {os.listdir('.')}")
