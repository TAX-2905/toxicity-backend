import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load model and vectorizer ONCE
model = joblib.load("toxicity_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

app = FastAPI(title="Kreol Toxicity API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class TextRequest(BaseModel):
    text: str

# Minimal preprocessing
def preprocess(text: str) -> str:
    return text.lower().strip()

@app.post("/predict")
def predict(request: TextRequest):
    text = preprocess(request.text)

    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]  # probability of toxic
    label = "toxic" if prob >= 0.5 else "non_toxic"

    return {
        "label": label,
        "confidence": round(float(prob), 3)
    }

@app.get("/")
def root():
    return {
        "message": "Kreol Toxicity API is running",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }
