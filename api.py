from fastapi import FastAPI

app = FastAPI(title="Churn Prediction API", version="1.0")

@app.get("/")
def root():
    return {"status": "API is running!"}

@app.get("/predict")
def predict(age: int = 35):
    return {"churn_prediction": 1 if age < 40 else 0}
