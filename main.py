import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Load model and vectorizer ONCE (important for performance)
# -------------------------
model = joblib.load("toxicity_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Kreol Toxicity API")

# -------------------------
# CORS configuration (FIXED)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",          # local dev
        "https://mltoxic.vercel.app",     # Vercel frontend (PRODUCTION)
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
# Minimal preprocessing
# -------------------------
def preprocess(text: str) -> str:
    return text.lower().strip()

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(request: TextRequest):
    text = preprocess(request.text)

    X = vectorizer.transform([text])
    prob = float(model.predict_proba(X)[0][1])  # probability of toxic
    label = "toxic" if prob >= 0.5 else "non_toxic"

    return {
        "label": label,
        "confidence": round(prob, 3)  # ALWAYS a number
    }

# -------------------------
# Health / root endpoint
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
