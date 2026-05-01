import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.preprocess import load_and_preprocess


def train_model():
    print("Loading and preprocessing data...")

    # Load dataset
    X, y = load_and_preprocess("dataset/health_data.csv")

    # Split data FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale AFTER split
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Load feature names
    with open("models/features.pkl", "rb") as f:
        feature_names = pickle.load(f)

    print("Training model...")

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure()
    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.tight_layout()

    os.makedirs("outputs/plots", exist_ok=True)

    plt.savefig("outputs/plots/feature_importance.png")
    plt.close()

    print("📊 Feature importance saved")

    # Save model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model saved")

    return model, X_test, y_test