# Titanic Survival Prediction API (End-to-End Project)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/FastAPI-0.100-green.svg)
![ML Library](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)
![Data](https://img.shields.io/badge/Pandas-2.0-blue.svg)

This repository contains the code for an end-to-end machine learning project that predicts passenger survival on the RMS Titanic. The project covers the complete data science lifecycle, from raw data processing to a deployed REST API.

---

## 1. 🚀 Project Objective

The goal is to build a complete ML system that:
1.  Cleans and processes raw, messy data.
2.  Trains and evaluates multiple classification models.
3.  Selects, analyzes, and interprets the best-performing model.
4.  Deploys the final model as a real-time REST API using FastAPI.

---

## 2. 🛠️ Technologies Used

* **Programming:** Python
* **Data Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Model Serialization:** Joblib
* **API Development:** FastAPI, Uvicorn
* **Data Visualization:** Matplotlib, Seaborn

---

## 3. 📂 Project Structure

. ├── app.py # FastAPI application script ├── titanic_prediction.ipynb # Jupyter Notebook for analysis and training ├── titanic_logistic_regression_model.pkl # Serialized (saved) model ├── requirements.txt # Project dependencies └── README.md # This file


---

## 4. 🚀 How to Use

### 4.1. Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-project-name.git](https://github.com/your-username/your-project-name.git)
    cd your-project-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file listing `fastapi`, `uvicorn`, `scikit-learn`, `joblib`, and `pandas`)*

### 4.2. Running the API

To start the live prediction server:

1.  **Run the Uvicorn server:**
    ```bash
    uvicorn app:app --reload
    ```
2.  The API will be live at `http://127.0.0.1:8000`.

### 4.3. API Endpoints

Once the server is running, go to the interactive documentation to test the API:
**`http://127.0.0.1:8000/docs`**

* **URL:** `/predict`
* **Method:** `POST`
* **Request Body (Example):**
    ```json
    {
      "Sex": 0,
      "Age": 0.475,
      "Fare": 0.139136,
      "Pclass_1": 1,
      "Pclass_2": 0,
      "Pclass_3": 0,
      "Family_size": 0.1,
      "Title_1": 1,
      "Title_2": 0,
      "Title_3": 0,
      "Title_4": 0,
      "Emb_1": 0,
      "Emb_2": 1,
      "Emb_3": 0
    }
    ```
* **Success Response (Example):**
    ```json
    {
      "Survival Prediction": "Survived"
    }
    ```

---

## 5. 📄 Resume / CV Project Description

*This section details how to describe the project in a professional resume.*

### Titanic Survival Prediction API (End-to-End Project)
*(Personal Project)*

#### Project Summary
Designed, trained, and deployed an end-to-end machine learning system to predict passenger survival on the Titanic, culminating in a production-ready REST API.

#### Data Preprocessing & Feature Engineering
Performed exploratory data analysis (EDA), imputed missing values (e.g., `Age`, `Embarked`), and **engineered new features** (e.g., `Title`, `FamilySize`) from raw text data to improve model performance.

#### Model Training & Evaluation
Trained and compared 3 classification models (Logistic Regression, Random Forest, Decision Tree), achieving a peak **accuracy of 88%** on the test set with Logistic Regression.

#### Model Interpretation
Analyzed confusion matrices to assess prediction errors and **leveraged Random Forest feature importance** to validate that the model learned logical patterns (e.g., `Sex`, `Fare`, and `Age` were top predictors).

#### API Deployment
Developed and **deployed the optimal model as a REST API** using **FastAPI** and Uvicorn, enabling real-time predictions from JSON payloads.

---

## 6. 📓 Jupyter Notebook Workflow

*This section details the professional structure of the `titanic_prediction.ipynb` notebook.*

### 1. Introduction & Problem Definition
* **1.1. Context:** Background of the Titanic disaster.
* **1.2. Project Objective:** Goal to build a predictive model.
* **1.3. Problem Type:** Defined as a Supervised, Binary Classification problem.
* **1.4. Success Metrics:** Primary metric (Accuracy) and secondary metrics (Confusion Matrix, F1-Score) are defined.

### 2. Data Preprocessing & Feature Engineering
*(This phase uses the **raw** data from Kaggle)*
* **2.1. Import Libraries:** Load all necessary packages.
* **2.2. Load Raw Data:** Load `train.csv` and `test.csv`.
* **2.3. Handle Missing Data:**
    * Impute `Age` using the median value.
    * Impute `Embarked` using the modal (most common) value.
    * Drop the `Cabin` column due to >70% missing data.
* **2.4. Feature Engineering:**
    * Create `FamilySize` from `SibSp` + `Parch`.
    * Extract `Title` (Mr., Mrs., etc.) from the `Name` column.
* **2.5. Data Transformation:**
    * Apply One-Hot Encoding to categorical features (`Sex`, `Embarked`, `Title`, `Pclass`).
    * Apply `MinMaxScaler` to continuous features (`Age`, `Fare`).
    * Drop original, unnecessary columns (`Name`, `Ticket`, etc.).

### 3. Model Training & Comparison
* **3.1. Define Features (X) and Target (y):** Split the processed data.
* **3.2. Model Selection:** Define the models to be compared (Logistic Regression, Random Forest, Decision Tree).
* **3.3. Train and Compare:** Loop through models, train each, and print its accuracy. A bar chart visualizes the comparison.
* **Result:** Logistic Regression is selected as the top performer (88% Accuracy).

### 4. In-Depth Model Analysis
* **4.1. Confusion Matrix:** Plot the confusion matrix for the selected Logistic Regression model to analyze its specific errors.
* **4.2. Feature Importance:** Use the trained Random Forest to plot a feature importance chart.
* **Analysis:** Confirm that `Sex`, `Fare`, and `Age` are the most significant predictors, validating the model's logic.

### 5. Model Serialization
* **5.1. Save Final Model:** The trained Logistic Regression model is saved to `titanic_logistic_regression_model.pkl` using `joblib`, making it ready for the API.
