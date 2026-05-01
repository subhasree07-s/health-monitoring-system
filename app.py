import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

# ===================== CONFIG =====================
st.set_page_config(page_title="Health Risk Predictor", layout="wide")

# ===================== LOAD MODEL =====================
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
feature_columns = pickle.load(open("models/features.pkl", "rb"))

# ===================== TITLE =====================
st.title("🏥 Health Risk Prediction System")
st.write("Enter your lifestyle details to predict your health risk")

# ===================== INPUT UI =====================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    sleep = st.number_input("Sleep Duration (hours)", 0.0, 12.0)
    quality = st.slider("Quality of Sleep (1-10)", 1, 10)
    activity = st.slider("Physical Activity Level", 0, 100)

with col2:
    stress = st.slider("Stress Level (1-10)", 1, 10)
    heart_rate = st.number_input("Heart Rate", 40, 120)
    steps = st.number_input("Daily Steps", 0, 20000)
    bmi_category = st.selectbox(
        "BMI Category",
        ["Underweight", "Normal", "Overweight", "Obese"]
    )

# ===================== PREPROCESS =====================
gender = 1 if gender == "Male" else 0

bmi_map = {
    "Underweight": 18,
    "Normal": 22,
    "Overweight": 27,
    "Obese": 32
}
bmi_value = bmi_map[bmi_category]

# ===================== BUTTON =====================
if st.button("Predict Risk"):

    # Prepare input
    input_dict = {
        "Age": age,
        "Gender": gender,
        "Sleep Duration": sleep,
        "Quality of Sleep": quality,
        "Physical Activity Level": activity,
        "Stress Level": stress,
        "Heart Rate": heart_rate,
        "Daily Steps": steps,
        "BMI_Value": bmi_value
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)

    # ===================== RESULT =====================
    st.subheader("🩺 Prediction Result")

    if prediction[0] == "High":
        st.error("⚠️ High Health Risk")
    elif prediction[0] == "Medium":
        st.warning("⚠️ Medium Health Risk")
    else:
        st.success("✅ Low Health Risk")

    st.write("Prediction:", prediction[0])

    # ===================== RISK SCORE =====================
    risk_score = int(np.max(proba) * 100)

    st.subheader("📊 Risk Score")
    st.progress(risk_score)
    st.write(f"Risk Score: {risk_score}/100")

    # ===================== RECOMMENDATIONS =====================
    st.subheader("💡 Recommendations")

    if prediction[0] == "High":
        st.write("- Reduce stress levels")
        st.write("- Improve sleep quality")
        st.write("- Increase physical activity")
    elif prediction[0] == "Medium":
        st.write("- Maintain balanced lifestyle")
    else:
        st.write("- Keep up your healthy lifestyle ✅")

    # ===================== FEATURE IMPORTANCE =====================
    st.subheader("📈 Feature Importance")

    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)

        fig, ax = plt.subplots()
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_columns[i] for i in indices])
        ax.set_title("Feature Importance")

        st.pyplot(fig)

    except:
        st.warning("Feature importance not available")

    # ===================== SHAP =====================
    st.subheader("🔍 Why this prediction?")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        # SAFE HANDLING
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                predicted_class_index = list(model.classes_).index(prediction[0])
                shap_vals = shap_values[predicted_class_index][0]
            else:
                shap_vals = shap_values[0][0]
        else:
            shap_vals = shap_values[0]

        shap_vals = np.array(shap_vals).flatten()
        shap_vals = shap_vals[:len(feature_columns)]

        shap_df = pd.DataFrame({
            "Feature": feature_columns,
            "Impact": shap_vals
        })

        shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)

        # Plot
        fig, ax = plt.subplots()
        colors = ["red" if v > 0 else "blue" for v in shap_df["Impact"]]

        ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors)
        ax.set_title("Feature Impact on Prediction")

        st.pyplot(fig)

        # Explanation
        st.subheader("🧠 Explanation")
        for _, row in shap_df.iterrows():
            if row["Impact"] > 0:
                st.write(f"🔴 {row['Feature']} increased risk")
            else:
                st.write(f"🔵 {row['Feature']} decreased risk")

    except Exception as e:
        st.warning("Explanation not available: " + str(e))