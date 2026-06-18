from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Initialize FastAPI
app = FastAPI(title="Churn Prediction API", version="1.0")

# Load and train model once at startup
df = pd.read_csv("data/churn.csv")
df = df.drop('customerID', axis=1)

le = LabelEncoder()
df['internet_service_encoded'] = le.fit_transform(df['internet_service'])
df = df.drop('internet_service', axis=1)

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Define input schema
class CustomerData(BaseModel):
    age: int
    contract_length: int
    monthly_charges: float
    total_charges: float
    internet_service: str

@app.get("/")
def root():
    return {"message": "✅ Churn Prediction API is running!"}

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        # Prepare input
        data = {
            'age': customer.age,
            'contract_length': customer.contract_length,
            'monthly_charges': customer.monthly_charges,
            'total_charges': customer.total_charges,
            'internet_service_encoded': le.transform([customer.internet_service])[0]
        }
        
        X_input = pd.DataFrame([data])
        X_scaled = scaler.transform(X_input)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        return {
            "customer_data": customer.dict(),
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "message": "⚠️ Customer likely to churn" if prediction == 1 else "✅ Customer likely to stay"
        }
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)