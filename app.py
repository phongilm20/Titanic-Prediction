import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title = "Titanic Survival Prediction")

try:
    model = joblib.load('titanic_logistic_regression_model.pkl')
except FileNotFoundError:
    model = None
    print("Error")


class Passenger(BaseModel):
    Sex: int
    Age: float
    Fare: float    
    Pclass_1: int 
    Pclass_2: int     
    Pclass_3: int
    Family_size:float
    Title_1: int  
    Title_2: int   
    Title_3: int   
    Title_4: int  
    Emb_1: int  
    Emb_2: int
    Emb_3: int
    class Config:
        schema_extra = {
            "example": {
                "Sex": 0, "Age": 0.475, "Fare": 0.139136, "Pclass_1": 1, "Pclass_2": 0,
                "Pclass_3": 0, "Family_size": 0.1, "Title_1": 1, "Title_2": 0, "Title_3": 0,
                "Title_4": 0, "Emb_1": 0, "Emb_2": 1, "Emb_3": 0
            }
        }

@app.get("/")
def read_root():
    return" Welcome to the Titanic Survival Prediction "

@app.post("/predict")
def predict_survival(passenger: Passenger):
    if model is None:
        return {"error"}
    

    features = [
        passenger.Sex,passenger.Age, passenger.Fare,
        passenger.Pclass_1 , passenger.Pclass_2, passenger.Pclass_3,
        passenger.Family_size,
        passenger.Title_1,passenger.Title_2,passenger.Title_3,passenger.Title_4,
        passenger.Emb_1,passenger.Emb_2,passenger.Emb_3      
    ]

    prediction_array = model.predict([features])
    prediction = int(prediction_array[0])
    if prediction == 1:
        return {"Survival Prediction: Survived"}
    else:
        return {"Survival Prediction: Not Survived"}

    
    



