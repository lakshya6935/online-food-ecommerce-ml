import streamlit as st
import joblib
import pandas as pd
import os

# Get the folder where this app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained ML models
delay_model = joblib.load(
    os.path.join(BASE_DIR, "delivery_delay_model.pkl")
)

rating_model = joblib.load(
    os.path.join(BASE_DIR, "service_rating_model.pkl")
)

st.title("Online Food / E-Commerce Delivery Prediction")
st.write("Two simple ML predictions from the supplied 100,000-row dataset.")

st.header("1. Delivery Delay Prediction")

minutes = st.number_input(
    "Delivery Time (Minutes)",
    min_value=1,
    max_value=200,
    value=35
)

if st.button("Predict Delay"):
    result = delay_model.predict(
        pd.DataFrame({
            "Delivery Time (Minutes)": [minutes]
        })
    )[0]

    st.success("Delivery Delay: " + str(result))


st.header("2. Service Rating Prediction")

feedback = st.selectbox(
    "Customer Feedback",
    [
        "Delivery person was rude.",
        "Easy to order, loved it!",
        "Excellent experience!",
        "Fast delivery, great service!",
        "Good quality products.",
        "Horrible experience, never ordering again.",
        "Items missing from order.",
        "Not fresh, disappointed.",
        "Packaging could be better.",
        "Quick and reliable!",
        "Very late delivery, not happy.",
        "Very satisfied with the service.",
        "Wrong item delivered."
    ]
)

if st.button("Predict Rating"):
    result = rating_model.predict(
        pd.DataFrame({
            "Customer Feedback": [feedback]
        })
    )[0]

    st.success(f"Predicted Service Rating: {result}/5")
