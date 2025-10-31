# Titanic Survival Prediction

## Overview
Predict Titanic passenger survival using machine learning models and a FastAPI web service.

## Dataset
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/azeembootwala/titanic)

## Features
- Data cleaning and advanced feature engineering
- Multiple model training (Logistic Regression, Random Forest, Decision Tree)
- Model evaluation (accuracy and confusion matrix)
- Prediction API via FastAPI

## Installation

### Requirements
- Python 3
- Packages: `fastapi`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `uvicorn`

### Install dependencies
pip install fastapi scikit-learn pandas numpy joblib uvicorn

## Usage

### 1. Train and Save Model
Train your models and save the best one as `titaniclogisticregressionmodel.pkl`.

### 2. Run API Service
uvicorn app:app --reload

### 3. Send Prediction Request
Send a POST request to `/predict` with the following sample payload:
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


## Model Evaluation

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 0.88     |
| Random Forest        | 0.85     |
| Decision Tree        | 0.80     |

- Confusion matrix and bar chart are available for detailed model performance comparison.

## Contributing

- Feel free to fork and make improvements: add models, improve features, or build a UI.
- Pull requests and feedback are welcome!
