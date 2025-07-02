import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

# === Load Soil CNN Model ===
cnn_model = tf.keras.models.load_model("soil_cnn_model.keras", compile=False)


with open("original_class_indices.json", "r") as f:
    original_indices = json.load(f)

with open("folder_to_target.json", "r") as f:
    folder_to_target = json.load(f)

index_to_class = {
    idx: folder_to_target[folder]
    for folder, idx in original_indices.items()
}

st.title("FarmBuddy")

st.header("Step 1: Upload Soil Image")
uploaded_file = st.file_uploader("Upload a soil image", type=["jpg", "jpeg", "png"])
predicted_soil_type = None

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    x = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = cnn_model.predict(x)
    pred_class_index = int(np.argmax(pred[0]))
    predicted_soil_type = index_to_class[pred_class_index]
    st.success(f"Detected Soil Type: **{predicted_soil_type}**")

# === Load Crop Prediction Model ===
@st.cache_resource
def load_crop_model():
    df = pd.read_csv("crop_yield.csv")
    X = df.drop(columns=["Crop", "Yield_tons_per_hectare"])
    y = df["Crop"]

    categorical_features = ["Region", "Soil_Type", "Weather_Condition"]
    numerical_features = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features)
    ], remainder="passthrough")

    model1 = LogisticRegression(max_iter=200)
    model2 = DecisionTreeClassifier(max_depth=5)
    model3 = KNeighborsClassifier(n_neighbors=5)

    ensemble = VotingClassifier(estimators=[
        ("lr", model1),
        ("dt", model2),
        ("knn", model3)
    ], voting="hard")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", ensemble)
    ])

    pipeline.fit(X, y)
    return pipeline

model = load_crop_model()

st.header("Step 2: Enter Environmental Details")

region = st.selectbox("Region", ["North", "South", "East", "West"])
weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy"])
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=2000.0, value=500.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
fertilizer = st.selectbox("Fertilizer Used", ["Yes", "No"])
irrigation = st.selectbox("Irrigation Used", ["Yes", "No"])
days_to_harvest = st.slider("Days to Harvest", min_value=60, max_value=180, value=120)

if st.button("Predict Crop"):
    user_input = pd.DataFrame([{
        "Region": region,
        "Soil_Type": predicted_soil_type,
        "Rainfall_mm": rainfall,
        "Temperature_Celsius": temperature,
        "Fertilizer_Used": fertilizer == "Yes",
        "Irrigation_Used": irrigation == "Yes",
        "Weather_Condition": weather,
        "Days_to_Harvest": days_to_harvest
    }])

    prediction = model.predict(user_input)[0]
    st.success(f"Recommended Crop: **{prediction}**")
