

from fastapi import FastAPI

from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

class CustomerData(BaseModel):
    Latitude: float
    Longitude: float
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Tenure_Months: int
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charges: float
    Total_Charges: float
    CLTV: int


@app.get("/")
def home():
    return {"message": "customer churn Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict")

def predict(data: CustomerData):
    input_df = pd.DataFrame([{
        "Latitude": data.Latitude,
        "Longitude": data.Longitude,
        "Gender": data.Gender,
        "Senior Citizen": data.Senior_Citizen,
        "Partner": data.Partner,
        "Dependents": data.Dependents,
        "Tenure Months": data.Tenure_Months,
        "Phone Service": data.Phone_Service,
        "Multiple Lines": data.Multiple_Lines,
        "Internet Service": data.Internet_Service,
        "Online Security": data.Online_Security,
        "Online Backup": data.Online_Backup,
        "Device Protection": data.Device_Protection,
        "Tech Support": data.Tech_Support,
        "Streaming TV": data.Streaming_TV,
        "Streaming Movies": data.Streaming_Movies,
        "Contract": data.Contract,
        "Paperless Billing": data.Paperless_Billing,
        "Payment Method": data.Payment_Method,
        "Monthly Charges": data.Monthly_Charges,
        "Total Charges": data.Total_Charges,
        "CLTV": data.CLTV
    }])

    categorical_cols = input_df.select_dtypes(include="object").columns
    input_df[categorical_cols] = input_df[categorical_cols].astype(str)

    input_processed = preprocessor.transform(input_df)

    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": round(float(probability),4)
    }
