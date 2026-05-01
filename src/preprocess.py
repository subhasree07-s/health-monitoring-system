import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import random
import os


def load_and_preprocess(path):
    df = pd.read_csv(path)

    # ===================== CLEANING =====================
    df = df.dropna()

    # ===================== ENCODING =====================
    if 'Gender' in df.columns:
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])

    # ===================== BMI MAPPING =====================
    bmi_map = {
        "Underweight": 18,
        "Normal": 22,
        "Overweight": 27,
        "Obese": 32
    }

    df['BMI_Value'] = df['BMI Category'].map(bmi_map)
    df['BMI_Value'] = df['BMI_Value'].fillna(22)

    # ===================== FEATURE ENGINEERING =====================
    def create_risk(row):
        score = 0

        # BMI
        if row['BMI_Value'] > 30:
            score += 2
        elif row['BMI_Value'] > 25:
            score += 1

        # Stress
        if row['Stress Level'] >= 8:
            score += 2
        elif row['Stress Level'] >= 5:
            score += 1

        # Sleep
        if row['Sleep Duration'] < 5:
            score += 2
        elif row['Sleep Duration'] < 7:
            score += 1

        # Activity
        if row['Physical Activity Level'] < 20:
            score += 2
        elif row['Physical Activity Level'] < 50:
            score += 1

        # ✅ STEP 1: assign to variable
        if score >= 5:
            risk = "High"
        elif score >= 3:
            risk = "Medium"
        else:
            risk = "Low"

        # ✅ STEP 2: ADD NOISE (INSIDE FUNCTION)
        if random.random() < 0.03:
            risk = random.choice(["Low", "Medium", "High"])

        # ✅ STEP 3: RETURN FINAL VALUE
        return risk

    # Apply risk creation
    df['Risk'] = df.apply(create_risk, axis=1)

    # ===================== DEBUG =====================
    print("\n📊 Risk Distribution:")
    print(df['Risk'].value_counts())

    # ===================== DROP UNUSED =====================
    df = df.drop([
        'Person ID',
        'Occupation',
        'BMI Category',
        'Blood Pressure',
        'Sleep Disorder'
    ], axis=1)

    # ===================== FEATURES & TARGET =====================
    X = df.drop('Risk', axis=1)
    y = df['Risk']

    # ===================== SAVE FEATURE ORDER =====================
    os.makedirs("models", exist_ok=True)

    with open("models/features.pkl", "wb") as f:
        pickle.dump(X.columns, f)

    return X, y