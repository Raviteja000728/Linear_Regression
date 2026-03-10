import streamlit as st
import pandas as pd

# scikit-learn is required for the prediction logic. In hosted environments
# such as Streamlit Cloud the package must be listed correctly in
# requirements.txt ("scikit-learn", not "sci-kit learn" etc.). If the import
# fails we show an error message and stop the app rather than crashing with a
# traceback like the one seen in the screenshot.
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
except ImportError:
    st.error("The scikit-learn library is not available. "
             "Make sure you have `scikit-learn` spelled correctly in your "
             "requirements.txt and redeploy the app.")
    st.stop()
import pickle
import os

# Function to load and train model
@st.cache_data
def load_and_train_model(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop_duplicates()
    X = df[["powerplay_wickets", "powerplay_score"]]
    y = df["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    return model, r2, mse

# Streamlit UI
st.title("Cricket Powerplay Score Predictor")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    model, r2, mse = load_and_train_model(uploaded_file)
    
    st.write(f"Model trained with R² Score: {r2:.4f}")
    st.write(f"Mean Squared Error: {mse:.4f}")
    
    st.header("Predict Final Score")
    
    powerplay_wickets = st.slider("Powerplay Wickets", min_value=0, max_value=10, value=2)
    powerplay_score = st.slider("Powerplay Score", min_value=0, max_value=100, value=50)
    
    if st.button("Predict"):
        # sklearn expects a 2‑D array for a single sample
        prediction = model.predict([[powerplay_wickets, powerplay_score]])
        st.write(f"Predicted Final Score: {prediction[0]:.2f}")
else:
    st.write("Please upload a CSV file to proceed.")
