import pickle
import numpy as np
import pandas as pd

def predict_risk(input_data):

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/features.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Scale input
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    return prediction[0]


if __name__ == "__main__":
    sample = [25, 1, 27, 6, 5, 70, 80, 5000]
    print("Predicted Risk:", predict_risk(sample))