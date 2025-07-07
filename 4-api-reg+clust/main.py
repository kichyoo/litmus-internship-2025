import pickle
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load regression model (pkl)
with open("best_regression_model.pkl", "rb") as f:
    regression_model = pickle.load(f)

# Load clustering model and preprocessor (joblib)
clustering_model = joblib.load("best_clustering_model.joblib")
preprocessor = joblib.load("data_preprocessor_clustering.joblib")

class UserInput(BaseModel):
    age: int
    gender: str

@app.post("/predict/")
def predict(input_data: UserInput):
    gender_map = {'Female': 0, 'Male': 1}
    gender_value = gender_map.get(input_data.gender.capitalize())
    if gender_value is None:
        return {"error": "Gender must be 'Male' or 'Female'"}

    input_df = pd.DataFrame([[input_data.age, gender_value]], columns=['Customer Age', 'Customer Gender'])
    predicted_value = regression_model.predict(input_df)[0]

    cluster_input = pd.DataFrame([{
        'Recency': 0,
        'Frequency': 1,
        'Monetary': predicted_value,
        'Age': input_data.age,
        'Customer Gender': input_data.gender.capitalize()
    }])
    cluster_processed = preprocessor.transform(cluster_input)
    cluster_label = clustering_model.predict(cluster_processed)[0]

    return {
        "predicted_order_value": round(predicted_value, 2),
        "assigned_cluster": int(cluster_label)
    }
