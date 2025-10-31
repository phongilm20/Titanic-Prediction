Titanic Survival Prediction
Overview
Predict Titanic passenger survival using machine learning models and API integration.

Dataset
Kaggle Titanic Dataset:
https://www.kaggle.com/datasets/azeembootwala/titanic

Features
Data cleaning and engineering

Multiple model training: Logistic Regression, Random Forest, Decision Tree

Model evaluation with accuracy and confusion matrix

FastAPI web service for predictions

Installation
bash
# Install required packages
pip install fastapi scikit-learn pandas numpy joblib uvicorn
Usage
Train models and save the best model (e.g., titaniclogisticregressionmodel.pkl).

Run API service:

bash
uvicorn app:app --reload
Send a POST request to /predict with passenger features:

json
{
  "Sex": 0,
  "Age": 0.475,
  "Fare": 0.139136,
  "Pclass1": 1,
  "Pclass2": 0,
  "Pclass3": 0,
  "Familysize": 0.1,
  "Title1": 1,
  "Title2": 0,
  "Title3": 0,
  "Title4": 0,
  "Emb1": 0,
  "Emb2": 1,
  "Emb3": 0
}
Evaluation
Model	Accuracy
Logistic Regression	0.88
Random Forest	0.85
Decision Tree	0.80
Contribution
Feel free to fork, improve data processing, try new models, or build a frontend!

