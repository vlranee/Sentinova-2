# sentinova/local_api.py
#
# CATATAN: file ini HANYA untuk debugging lokal manual, mis. lewat
# `uvicorn local_api:app --reload` lalu tes endpoint /predict pakai curl
# atau file HTML `sentinova.html` yang lama (versi berbasis fetch()).
# app.py (yang di-deploy) TIDAK memakai file ini lagi — sejak app.py
# dipindah ke pendekatan native Streamlit custom component
# (component_frontend/index.html), prediksi dipanggil langsung sebagai
# fungsi Python tanpa HTTP sama sekali. File ini dibiarkan ada sebagai
# alat bantu development saja.
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