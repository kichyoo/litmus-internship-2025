from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load saved model and encoders
model = joblib.load("best_model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_category = joblib.load("le_category.pkl")

app = FastAPI()

class UserInput(BaseModel):
    age: int
    gender: str  # 'Male' or 'Female'

@app.post("/recommend")
def recommend_category(input: UserInput):
    try:
        gender_clean = input.gender.strip().capitalize()
        gender_encoded = le_gender.transform([gender_clean])[0]
        features = pd.DataFrame([[input.age, gender_encoded]], columns=["Customer Age", "Customer Gender"])
        prediction = model.predict(features)
        category = le_category.inverse_transform(prediction)[0]
        return {"recommended_category": category}
    except Exception as e:
        return {"error": str(e)}
