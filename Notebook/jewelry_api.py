
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

MODEL_URI = "runs:/e649349d5b7f49a09451f1965123c20a/model"
model = mlflow.pyfunc.load_model(MODEL_URI)

class ProductInput(BaseModel):
    Quantity_of_SKU: float
    Category_alias: str
    Brand_ID: float
    Product_gender: str
    Main_Color: str
    Main_metal: str
    Main_gem: str

app = FastAPI(title="Jewelry Price Prediction API")

@app.post("/predict")
def predict_price(input_data: ProductInput):
    df = pd.DataFrame([input_data.dict()])
    df.columns = [
        "Quantity of SKU in the order",
        "Category alias", "Brand ID", "Product gender",
        "Main Color", "Main metal", "Main gem"
    ]
    prediction = model.predict(df)[0]
    return {"predicted_price_usd": round(prediction, 2)}
