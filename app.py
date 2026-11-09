
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "insurance_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_names.pkl"


# Load saved model and features
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURE_PATH, "rb") as f:
    feature_names = pickle.load(f)

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Prediction",
    page_icon="🏥",
    layout="centered"
)

# Application title
st.title("🏥 Medical Insurance Cost Prediction")

st.write(
    "Enter your personal details to estimate your medical insurance charges."
)

# Input form
with st.form("insurance_form"):

    st.subheader("Personal Details")
    name = st.text_input("Enter your name")

    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Gender",
        ["female", "male"]
    )

    bmi = st.slider(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoker = st.selectbox(
        "Smoker",
        ["no", "yes"]
    )

    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

    submit = st.form_submit_button("Predict Insurance Cost")

# Prediction
if submit:

    input_data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": int(sex == "male"),
        "smoker_yes": int(smoker == "yes"),
        "region_northwest": int(region == "northwest"),
        "region_southeast": int(region == "southeast"),
        "region_southwest": int(region == "southwest")
    }

    input_df = pd.DataFrame([input_data])

    # Match training feature order
    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    prediction = model.predict(input_df)[0]

    st.success("Prediction Completed!")
    st.subheader(f"Hello, {name}! 👋")

    st.subheader("Estimated Insurance Charges")

    st.metric(
    "Estimated Insurance Charges",
    f"₹{prediction:,.2f}"
)

    st.info(
         f"Dear {name}, this is your estimated insurance cost, "
    "not an actual insurance quotation."
    )
