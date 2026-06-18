from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn using ML",
    version="1.0"
)

df = pd.read_csv("data/churn.csv")
df = df.drop('customerID', axis=1)

le = LabelEncoder()
df['internet_service_encoded'] = le.fit_transform(df['internet_service'])
df = df.drop('internet_service', axis=1)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_test_scaled, y_test)

class CustomerInput(BaseModel):
    age: int
    contract_length: int
    monthly_charges: float
    total_charges: float
    internet_service: str

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    message: str

@app.get("/")
def root():
    return {"status": "API running", "accuracy": round(accuracy, 4)}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        internet_encoded = le.transform([customer.internet_service])[0]
        data = np.array([[customer.age, customer.contract_length, customer.monthly_charges, customer.total_charges, internet_encoded]])
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0][1]
        message = "Customer will churn" if prediction == 1 else "Customer will stay"
        return PredictionResponse(churn_prediction=int(prediction), churn_probability=round(float(probability), 4), message=message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
