import joblib
import pandas as pd
from fastapi import FastAPI
from src.schema.schema import LoanData

app = FastAPI()

# load trained model
model = joblib.load("models/model.pkl")


@app.get("/")
def home():

    return {"message": "Loan Prediction API Running"}



@app.post("/predict")
def predict(data: LoanData):

    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return {"prediction": result}