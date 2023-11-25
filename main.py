import pickle
from math import nan

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


def remove_postfix(val):
    if isinstance(val, str):
        try:
            return float(val.split()[0])
        except:
            return nan

    return val


def preprocess(df):
    for col in ["mileage", "engine", "max_power"]:
        df[col] = df[col].apply(remove_postfix)
    
    df = df.drop(columns=["torque"])
    
    medians = {
        'mileage': 19.369999999999997,
        'engine': 1248.0,
        'max_power': 81.86,
        'seats': 5.0
    }
    for col, median in medians.items():
        df[col] = df[col].fillna(median)
    
    df["engine"] = df["engine"].astype(int)
    df["seats"] = df["seats"].astype(int)

    df = df.drop(columns=["name", "selling_price"])
    
    cat_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    return np.concatenate([
        df.drop(columns=cat_columns).to_numpy(),
        ohe.transform(df[cat_columns]),
    ], axis=1)


with open("best_model.pkl", "rb") as file:
    model_dict = pickle.load(file)
    model = model_dict["model"]
    ohe = model_dict["ohe"]

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
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


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return model.predict(preprocess(pd.DataFrame([item.model_dump()])))


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df = pd.DataFrame([item.model_dump() for item in items])
    df_preprocessed = preprocess(df)
    return model.predict(df_preprocessed)
