Health Monitoring & Risk Prediction System

An Explainable Machine Learning system that predicts health risk levels based on lifestyle and physiological data, with real-time predictions and interpretable insights using SHAP.


Overview

Lifestyle-related diseases such as obesity, cardiovascular issues, and sleep disorders are increasing due to unhealthy habits. Traditional healthcare systems rely on expensive and late-stage diagnostics.

This project provides a preventive healthcare solution by predicting risk levels using simple lifestyle inputs like sleep, stress, and physical activity. 


Key Features

* Predicts **Low / Medium / High health risk**
* Real-time prediction using **Streamlit web app**
* Explainable AI using **SHAP (feature contribution)**
* Visual outputs (feature importance, risk score, graphs)
* Model evaluation with accuracy, precision, recall, F1-score
* Personalized health recommendations


Machine Learning Approach

* Algorithm: Random Forest Classifier
* Type: Supervised Learning
* Labels: Generated using rule-based scoring system 


Model Performance

* Accuracy: **93.54%**
* Precision: **94.19%**
* Recall: **93.54%**
* F1 Score: **93.42%** 


System Architecture

The system follows a structured pipeline:

1. Data Collection (User + Dataset)
2. Data Preprocessing (Cleaning, Encoding, Normalization)
3. Feature Engineering (Risk label generation)
4. Model Training (Random Forest)
5. Evaluation (Accuracy, Precision, Recall, F1)
6. Explainability (SHAP)
7. Deployment (Streamlit App)


Dataset

* Source: Kaggle

* Type: Tabular Data

* Features:

  * Age, Gender
  * Sleep Duration
  * Stress Level
  * Physical Activity
  * BMI
  * Heart Rate
  * Daily Steps

* Size: ~300–400 records


Project Structure

health-monitoring-system/
│
├── dataset/
├── models/
├── outputs/
│   └── plots/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── feature_engineering.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md

Screenshots

* Prediction Interface

![Prediction Interface]("https://github.com/user-attachments/assets/c9c6bdf7-0124-4fca-95cf-21231574a7c4")

* Feature Importance

![Feature Importance]("https://github.com/user-attachments/assets/40f21cf0-8a59-438e-a1b7-8e801c7ebb35")

* SHAP Explanation

![SHAP Explanation]("https://github.com/user-attachments/assets/a4e3e627-e651-4390-80f9-6645b2135b4a")


## ▶️ How to Run the Project

1. Install Dependencies

pip install -r requirements.txt

2. Run Application

streamlit run app.py


Applications

* Personal health monitoring
* Fitness & wellness apps
* Telemedicine platforms
* Corporate wellness programs
* Public health awareness systems 


Limitations

* Dataset is relatively small
* Risk labels are rule-based (not clinically validated)
* Depends on user input accuracy 


Future Improvements

* Use larger real-world datasets
* Integrate wearable/real-time health data
* Improve model with deep learning
* Deploy on cloud for public access


Conclusion

This project demonstrates how **machine learning + explainable AI** can be used for **early health risk detection**, enabling users to make better lifestyle decisions and promoting preventive healthcare. 



