
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
import io
import tempfile
from fastapi.responses import FileResponse

app = FastAPI()

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def preprocess_data(df):
    df_encoded = pd.get_dummies(df)
    missing_cols = set(feature_columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns]
    df_scaled = scaler.transform(df_encoded)
    return df_scaled

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_dict = item.dict()
    df = pd.DataFrame([item_dict])
    df_scaled = preprocess_data(df)
    prediction_log = model.predict(df_scaled)
    prediction = np.expm1(prediction_log)
    return prediction[0]

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    items_data = [item.dict() for item in items]
    df = pd.DataFrame(items_data)
    df_scaled = preprocess_data(df)
    predictions_log = model.predict(df_scaled)
    predictions = np.expm1(predictions_log)
    return predictions.tolist()

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    df_scaled = preprocess_data(df)
    predictions_log = model.predict(df_scaled)
    predictions = np.expm1(predictions_log)
    df['prediction'] = predictions
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    return FileResponse(tmp_path, media_type='text/csv', filename='predictions.csv')
