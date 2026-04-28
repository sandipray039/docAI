# DocAI — Local RAG over PDFs (100% Free)

Chat with your PDF documents using:

| Component | Tech | Cost |
|---|---|---|
| LLM | [Groq](https://console.groq.com) (Llama 3) | Free |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Free, runs on CPU |
| Vector store | ChromaDB (local) | Free |
| Backend | FastAPI | Free |
| Frontend | Streamlit | Free |
| Deploy (backend) | [Render](https://render.com) free tier | Free |
| Deploy (frontend) | [Streamlit Community Cloud](https://share.streamlit.io) | Free |

## Architecture

```
Streamlit (8501) ──HTTP──▶ FastAPI (8000) ──▶ ChromaDB (local disk)
                                       └─────▶ Groq API (LLM, free)
                            sentence-transformers (CPU embeddings, local)
```

---

## Quick start (local)

### 1. Get a free Groq API key

Go to https://console.groq.com → sign up (no credit card) → create API key.

### 2. Set up your `.env`

```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### 3. Run the backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

First run downloads the embedding model (~90 MB) automatically. Then visit:
- API docs: http://localhost:8000/docs
- Health:   http://localhost:8000/api/health

### 4. Run the frontend

Open a **second terminal**:

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at http://localhost:8501.

### 5. Test it

Upload a PDF using the sidebar, then ask questions in the chat.

Or via curl:
```bash
# Upload
curl -F "file=@yourfile.pdf" http://localhost:8000/api/upload

# Query
curl -X POST http://localhost:8000/api/query \
  -H "content-type: application/json" \
  -d "{\"question\": \"What is this document about?\"}"
```

---

## Deploy for free

### Backend → Render

1. Push this repo to GitHub.
2. Go to https://render.com → New → Web Service → connect your repo.
3. Set **Root Directory** to `backend`.
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `GROQ_API_KEY` = your key.
7. Add a **Disk** (free 1 GB) mounted at `/app/data` so the index persists.
8. Deploy.

Note the backend URL (e.g. `https://docai-backend.onrender.com`).

### Frontend → Streamlit Community Cloud

1. Push your repo to GitHub (the `frontend/` folder must be included).
2. Go to https://share.streamlit.io → New app.
3. Repo: your GitHub repo, Branch: `main`, Main file: `frontend/streamlit_app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   API_URL = "https://docai-backend.onrender.com/api"
   ```
5. Deploy.

Done — fully free, publicly accessible.

---

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload + ingest a PDF |
| POST | `/api/query` | `{question, top_k?, source?}` → answer + sources |
| POST | `/api/query/stream` | Streaming token response |
| GET | `/api/documents` | List indexed files |
| DELETE | `/api/documents/{source}` | Remove a file from index |
| POST | `/api/reset` | Clear entire index |

---

## Free Groq models you can use

Set `LLM_MODEL` in `.env` to any of:

| Model | Context | Speed |
|---|---|---|
| `llama3-8b-8192` | 8k | Very fast (default) |
| `llama3-70b-8192` | 8k | Smarter, still fast |
| `mixtral-8x7b-32768` | 32k | Large context |
| `gemma2-9b-it` | 8k | Good alternative |

---

## Project layout

```
backend/
  app/
    main.py              # FastAPI app + CORS
    config.py            # env-driven settings (pydantic-settings)
    api/routes.py        # HTTP endpoints
    services/
      pdf_service.py     # extract + chunk PDFs
      vector_service.py  # ChromaDB + sentence-transformers
      llm_service.py     # Groq chat (sync + streaming)
      rag_service.py     # ingest + answer orchestration
  requirements.txt
  render.yaml            # Render deploy config
frontend/
  streamlit_app.py
  requirements.txt
.env.example
```
