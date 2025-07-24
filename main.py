from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import pickle
from pydantic import BaseModel, Field

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

with open("model/catboost_classifier.pkl", "rb") as f:
    model = pickle.load(f)

def predict_gender(new_data):
    """
    new_data: pd.DataFrame with columns ['meanfun', 'IQR', 'Q25', 'sd', 'sp.ent']
    Returns predicted gender labels (0=female, 1=male)
    """
    preds = model.predict(new_data)
    return preds

class VoiceInput(BaseModel):
    meanfun: float = Field(..., ge=0.055565, le=0.237636)
    IQR: float = Field(..., ge=0.014558, le=0.252225)
    Q25: float = Field(..., ge=0.000229, le=0.247347)
    sd: float = Field(..., ge=0.018363, le=0.115273)
    sp_ent: float = Field(..., ge=0.738651, le=0.981997, alias="sp.ent")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    meanfun: float = Form(...),
    IQR: float = Form(...),
    Q25: float = Form(...),
    sd: float = Form(...),
    sp_ent: float = Form(..., alias="sp.ent")
):
    input_data = pd.DataFrame({
        'meanfun': [meanfun],
        'IQR': [IQR],
        'Q25': [Q25],
        'sd': [sd],
        'sp.ent': [sp_ent]
    })
    
    pred = predict_gender(input_data)
    gender_map = {0: "Female", 1: "Male"}
    predicted_gender = gender_map[pred[0]]
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": predicted_gender
        }
    )