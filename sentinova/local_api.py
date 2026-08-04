# sentinova/local_api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.model_loader import load_model, predict
from utils.preprocessing import preprocess_text

MODEL_NAME = "rantirann/sentinova-indobert"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

print("Loading model, tunggu sebentar...")
tokenizer, model, cfg = load_model(MODEL_NAME)
print("Model berhasil dimuat!")

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    clean = preprocess_text(req.text)
    label, confidence, all_scores = predict(clean, tokenizer, model, cfg)
    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "scores": {k: round(v, 4) for k, v in all_scores.items()},
    }