import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Load model and vectorizer ONCE
# -------------------------
model = joblib.load("toxicity_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

import os
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Kreol Toxicity API")

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://mltoxic.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request schema
# -------------------------
class TextRequest(BaseModel):
    text: str

# -------------------------
# Preprocess
# -------------------------
def preprocess(text: str) -> str:
    return text.lower().strip()

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(request: TextRequest):
    text = request.text.lower().strip()

    X = vectorizer.transform([text])
    prob = float(model.predict_proba(X)[0][1])
    is_toxic = prob >= 0.5

    try:
        supabase.table("search_history").insert({
            "search_text": text,
            "is_toxic": is_toxic
        }).execute()
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

    return {
        "status": "success",
        "label": "toxic" if is_toxic else "non_toxic",
        "confidence": round(prob, 3)
    }




# -------------------------
# Root endpoint
# -------------------------
@app.get("/")
def root():
    return {
        "message": "Kreol Toxicity API is running",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }
