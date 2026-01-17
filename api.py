import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and vectorizer ONCE
model = joblib.load("model/toxicity_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

app = FastAPI(title="Kreol Toxicity API")

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
