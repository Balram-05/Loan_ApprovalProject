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
    print(input_data)

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    return {
        "prediction": "Loan Approved" if prediction[0] == 1 else "Loan Rejected",
        "approval_probability": float(prob[0][1])
    }