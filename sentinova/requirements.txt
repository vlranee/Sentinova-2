streamlit>=1.53.0
transformers>=4.47.0
torch>=2.5.0
numpy>=2.0.0
huggingface_hub>=0.27.0

# Opsional — hanya dipakai kalau menjalankan local_api.py secara terpisah
# untuk debugging lokal via `uvicorn local_api:app`. Tidak dipakai lagi oleh
# app.py (production/deploy) sejak pindah ke custom component (lihat app.py).
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.10.0
requests>=2.32.0
