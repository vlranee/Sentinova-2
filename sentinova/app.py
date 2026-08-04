import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="Sentinova",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding: 0 !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
[data-testid="stSidebar"] { display: none; }
iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model (sekali saja, di-cache) ─────────────────────────────────────
MODEL_NAME = "rantirann/sentinova-indobert"

@st.cache_resource(show_spinner="Memuat model IndoBERT…")
def load_resources():
    from utils.model_loader import load_model
    return load_model(MODEL_NAME)

tokenizer, model, cfg = load_resources()

from utils.model_loader import predict
from utils.preprocessing import preprocess_text


def run_prediction(text: str):
    clean = preprocess_text(text)
    label, confidence, all_scores = predict(clean, tokenizer, model, cfg)
    return label, confidence, all_scores


# ─────────────────────────────────────────────────────────────────────────
# Custom component (native Streamlit, bidirectional) — MENGGANTIKAN
# pendekatan FastAPI/Mount + `streamlit.starlette.App` yang sebelumnya
# dipakai. Tidak ada HTTP call sama sekali: UI (component_frontend/index.html)
# mengirim nilai lewat `Streamlit.setComponentValue()` (protokol postMessage
# bawaan komponen Streamlit), Python di sini menerimanya sebagai
# `component_value`, menjalankan predict() langsung di proses yang sama,
# lalu mengirim hasilnya balik lewat argumen `result` / `batch_result`.
# Ini fitur custom component yang sudah stabil bertahun-tahun (bukan
# eksperimental), jadi tidak tergantung dukungan ASGI/Starlette dari
# platform hosting.
# ─────────────────────────────────────────────────────────────────────────
_COMPONENT_DIR = Path(__file__).parent / "component_frontend"
sentinova_ui = components.declare_component("sentinova_ui", path=str(_COMPONENT_DIR))

if "last_request_id" not in st.session_state:
    st.session_state.last_request_id = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_batch_result" not in st.session_state:
    st.session_state.last_batch_result = None

component_value = sentinova_ui(
    result=st.session_state.last_result,
    batch_result=st.session_state.last_batch_result,
    key="sentinova_main",
)

# Proses request baru dari frontend (kalau ada dan belum pernah diproses).
if component_value and component_value.get("requestId") != st.session_state.last_request_id:
    st.session_state.last_request_id = component_value["requestId"]
    action = component_value.get("action")

    if action == "predict":
        text = component_value.get("text", "")
        try:
            label, confidence, _scores = run_prediction(text)
            st.session_state.last_result = {
                "requestId": component_value["requestId"],
                "prediction": label,
                "confidence": round(confidence, 4),
            }
        except Exception as e:
            st.session_state.last_result = {
                "requestId": component_value["requestId"],
                "error": str(e),
            }
        st.session_state.last_batch_result = None
        st.rerun()

    elif action == "predict_batch":
        texts = component_value.get("texts", [])
        results = []
        for text in texts:
            try:
                label, confidence, _scores = run_prediction(text)
                results.append({
                    "text": text,
                    "prediction": label,
                    "confidence": round(confidence, 4),
                })
            except Exception:
                results.append({"text": text, "error": True})
        st.session_state.last_batch_result = {
            "requestId": component_value["requestId"],
            "results": results,
        }
        st.session_state.last_result = None
        st.rerun()
