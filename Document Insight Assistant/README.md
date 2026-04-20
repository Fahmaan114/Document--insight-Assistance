# Document Insight Assistant

Document Insight Assistant is a modular private document question-answering application built as a portfolio project. The target architecture uses FastAPI for the backend, Streamlit for the frontend, OpenAI APIs for model integrations, and FAISS for local vector search.

## Current Scope

The current MVP supports end-to-end local document Q&A:

- PDF and TXT upload ingestion
- Text cleaning and overlapping chunk generation
- configurable local or OpenAI-backed embeddings
- local FAISS indexing with metadata persistence
- local index status and reset controls for demos
- retrieval with a no-results fallback
- grounded answer generation with source snippets
- Streamlit frontend connected to the backend upload and answer endpoints
- deployment-ready configuration for public backend and frontend hosting

Out of scope for the current MVP:

- authentication
- web retrieval
- multi-user document isolation
- production deployment hardening

## Project Structure

```text
.
+-- app/
|   +-- __init__.py
|   +-- config.py
|   +-- main.py
|   `-- routes.py
+-- data/
|   `-- vectorstore/
|       `-- .gitkeep
+-- frontend/
|   +-- __init__.py
|   +-- api_client.py
|   `-- streamlit_app.py
+-- rag/
|   +-- __init__.py
|   +-- chunking.py
|   +-- embeddings.py
|   +-- ingest.py
|   +-- prompt_builder.py
|   `-- retrieve.py
+-- tests/
|   `-- .gitkeep
+-- .env.example
+-- .gitignore
+-- README.md
+-- render.yaml
`-- requirements.txt
```

## Setup

1. Create a virtual environment from the project root:

```bash
python -m venv .venv
```

2. Activate it.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and fill in your values.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

## Run The Backend

Run from the project root:

```bash
python -m uvicorn app.main:app --reload
```

The API will start locally, and the health endpoint will be available at:

- `http://127.0.0.1:8000/health`

The upload endpoint is available at:

- `POST http://127.0.0.1:8000/upload`

The retrieval endpoint is available at:

- `POST http://127.0.0.1:8000/retrieve`

The grounded answer endpoint is available at:

- `POST http://127.0.0.1:8000/answer`

The index status endpoint is available at:

- `GET http://127.0.0.1:8000/index-status`

The reset endpoint is available at:

- `DELETE http://127.0.0.1:8000/index`

## Run The Frontend

Run from the project root in a separate terminal:

```bash
python -m streamlit run frontend/streamlit_app.py
```

The Streamlit app will start locally, usually at:

- `http://localhost:8501`

By default, the frontend connects to `http://127.0.0.1:8000`. If your backend runs elsewhere, set `BACKEND_BASE_URL` in `.env` or update it in the Streamlit sidebar.

## GitHub + Render Deployment

This repository is ready for a free frontend deployment on Streamlit Community Cloud while keeping the backend separate.

Deployment shape:

- GitHub hosts the source repository
- Streamlit Community Cloud hosts the frontend
- a separately hosted FastAPI backend remains responsible for upload, indexing, retrieval, answering, and persistence

The frontend entry file is:

- `frontend/streamlit_app.py`

The deployed frontend resolves the backend URL in this order:

1. `st.secrets["BACKEND_BASE_URL"]`
2. `BACKEND_BASE_URL` environment config
3. local `.env` / default settings for development

## Streamlit Frontend Requirements

The existing [requirements.txt](</C:/Users/fahman/Dropbox/PC/Downloads/Document Insight Assistant/requirements.txt>) is sufficient for Streamlit Community Cloud deployment. No extra deployment-only package changes are required for this MVP.

## Backend URL Configuration

For local development:

- `BACKEND_BASE_URL=http://127.0.0.1:8000`

For Streamlit Community Cloud:

- add `BACKEND_BASE_URL` in Streamlit Secrets
- point it to your public backend URL

Example:

```toml
BACKEND_BASE_URL = "https://your-backend.example.com"
```

The deployed frontend should not depend on localhost.

## Deploy The Frontend On Streamlit Community Cloud

1. Push this project to GitHub.

2. Open [Streamlit Community Cloud](https://share.streamlit.io/).

3. Click `Create app`.

4. Select your GitHub repository.

5. Choose the branch you want to deploy.

6. Set the main file path to:

```text
frontend/streamlit_app.py
```

7. Open `Advanced settings`.

8. Set Python version to:

```text
3.11
```

9. In `Secrets`, add:

```toml
BACKEND_BASE_URL = "https://your-backend.example.com"
```

10. Deploy the app.

11. After deployment, verify:

- the app loads
- upload works
- answering works
- supporting excerpts appear
- `Start over` still resets the indexed backend data

## Backend Notes For Frontend Deployment

- The backend must already be reachable from the public internet.
- The backend must expose `GET /health`, `POST /upload`, `POST /answer`, `GET /index-status`, and `DELETE /index`.
- The frontend does not store FAISS data itself; persistence remains entirely a backend concern.
- If your backend URL changes, update `BACKEND_BASE_URL` in Streamlit Secrets and redeploy.

## Production Notes

- The frontend remains a Streamlit single-page app.
- The current UI and product flow are unchanged.
- The backend URL stays configurable without hardcoding localhost in deployed mode.
- The backend architecture remains unchanged, including FAISS persistence and grounded answering.

## Local Development After Deployment Changes

Local development still works the same way:

1. Copy `.env.example` to `.env`
2. Keep `BACKEND_BASE_URL=http://127.0.0.1:8000`
3. Keep `VECTORSTORE_DIR=data/vectorstore`
4. Optionally leave `ALLOWED_ORIGINS` blank for local-only usage
5. Run the backend:

```bash
python -m uvicorn app.main:app --reload
```

6. Run the frontend in a second terminal:

```bash
python -m streamlit run frontend/streamlit_app.py
```

## Sharing The App

Once deployed:

- share the Streamlit frontend public URL
- keep the backend URL available for health checks and debugging
- if the backend URL changes, update Streamlit Secrets with the new `BACKEND_BASE_URL`
- if your backend enforces origin rules, allow the deployed Streamlit frontend origin there

## End-To-End Usage

1. Start the FastAPI backend:

```bash
python -m uvicorn app.main:app --reload
```

2. In a second terminal, start the Streamlit frontend:

```bash
python -m streamlit run frontend/streamlit_app.py
```

3. Open the Streamlit app in your browser.

4. Confirm the top status panel shows whether the local index is empty or already populated.

5. Upload a local `PDF` or `TXT` file.

6. Ask a question about the uploaded documents.

7. Review the grounded answer and the source snippets shown beneath it.

8. Use the reset control in the Streamlit UI when you want to clear the local FAISS index and metadata for a fresh demo.

## Health Check Response

`GET /health` returns a minimal readiness payload:

```json
{
  "status": "healthy",
  "app_name": "Document Insight Assistant",
  "environment": "development"
}
```

## Upload Response Shape

`POST /upload` accepts a single `PDF` or `TXT` file as multipart form data and returns the chunked document plus local indexing metadata.

Example response:

```json
{
  "filename": "sample.txt",
  "source_type": "txt",
  "chunk_count": 2,
  "indexing": {
    "embedding_provider": "local_dummy",
    "embedding_model": "local_dummy_384",
    "embedding_dimension": 384,
    "index_path": "/abs/path/to/data/vectorstore/chunks.faiss",
    "metadata_path": "/abs/path/to/data/vectorstore/chunks_metadata.json",
    "added_vector_count": 2,
    "total_vector_count": 8
  },
  "chunks": [
    {
      "filename": "sample.txt",
      "chunk_id": "sample-a1b2c3d4e5-p0000-c0001",
      "chunk_text": "First chunk text...",
      "source_type": "txt",
      "page_number": null
    }
  ]
}
```

## Retrieval Response Shape

`POST /retrieve` accepts a question and returns the top-k ranked chunks from the persisted local vector store.

Example request:

```json
{
  "question": "What does the contract say about termination?",
  "top_k": 3
}
```

## Answer Response Shape

`POST /answer` retrieves relevant chunks first, then generates an answer strictly from that retrieved context.

Example request:

```json
{
  "question": "What does the contract say about termination?",
  "top_k": 3
}
```

Example response:

```json
{
  "question": "What does the contract say about termination?",
  "answer": "Termination may occur with thirty days written notice.",
  "answer_supported": true,
  "message": "Answer generated strictly from retrieved document context.",
  "sources": [
    {
      "filename": "contract.pdf",
      "chunk_id": "contract-abc123def0-p0004-c0007",
      "source_type": "pdf",
      "page_number": 4,
      "rank": 1,
      "score": 0.8123,
      "snippet": "Termination may occur with thirty days written notice."
    }
  ]
}
```

Example response:

```json
{
  "question": "What does the contract say about termination?",
  "top_k": 3,
  "no_results": false,
  "message": "Retrieved 3 relevant chunk(s).",
  "results": [
    {
      "filename": "contract.pdf",
      "chunk_id": "contract-abc123def0-p0004-c0007",
      "chunk_text": "Termination may occur with thirty days written notice...",
      "source_type": "pdf",
      "page_number": 4,
      "vector_id": 6,
      "rank": 1,
      "score": 0.8123
    }
  ]
}
```

## Demo Notes

- The Streamlit app is a single-page local demo. It is meant to show upload, indexing, grounded answering, source snippets, and reset flow without requiring Swagger.
- The local vector store is shared for the current project workspace. Uploading multiple files appends their chunks into the same FAISS index until you reset it.
- Resetting from the UI deletes the persisted `chunks.faiss` and `chunks_metadata.json` artifacts together, so the next question run behaves like no documents are indexed.
- If no documents are indexed, the UI will prompt you to upload a file before asking questions.
- Unsupported questions stay grounded: the app will explain that the answer could not be supported by the uploaded document context rather than guessing.

## Known Limitations

- The frontend can be hosted for free on Streamlit Community Cloud, but the backend still must be hosted separately and stay publicly reachable.
- FAISS index files and metadata persistence remain entirely on the backend host, not on Streamlit Community Cloud.
- If the backend runs on a temporary or free host, uploaded data and indexes may be lost on restart, sleep, or redeploy.
- The frontend experience depends on backend uptime, backend storage, and network reachability.

## Notes

- The application is structured so later phases can add ingestion, chunking, embeddings, retrieval, and grounded answer generation without reshaping the project.
- `data/vectorstore/` and `tests/` are intentionally present in the repository scaffold, even before they contain implementation files.
- Default chunking uses `1000` characters with `200` characters of overlap, which keeps chunks focused while preserving context at chunk boundaries.
- `page_number` is 1-based for PDF chunks and `null` for TXT chunks.
- `chunk_id` values include a safe document key, a page marker, and a chunk sequence so they remain traceable during later multi-file indexing.
- The local vector store uses FAISS plus a JSON metadata sidecar. Vector row `i` in FAISS maps to metadata entry `vector_id == i`, so vector-to-metadata alignment stays stable across save and reload.
- `EMBEDDING_PROVIDER=local` is the default development path. It prefers `sentence-transformers` with `LOCAL_EMBEDDING_MODEL`, and in development it falls back to deterministic `local_dummy` embeddings when that local model path is unavailable.
- `EMBEDDING_PROVIDER=local_dummy` is the simplest no-cost development mode and creates deterministic local embeddings without any external API usage.
- `EMBEDDING_PROVIDER=openai` keeps the original OpenAI path available. When you use it, set `OPENAI_API_KEY` and `OPENAI_EMBEDDING_MODEL`.
- Retrieval uses the same active embedding provider as indexing. If the current provider or model does not match the persisted FAISS store, the API returns a clear conflict error instead of mixing incompatible vectors.
- Retrieval applies a configurable usefulness threshold via `RETRIEVAL_MIN_SCORE` and deduplicates results by `chunk_id` before returning them. With the default threshold, obviously unrelated questions are more likely to return a clean no-useful-context fallback instead of weak matches.
- `ANSWER_PROVIDER=local` is the default grounded-answer path and produces extractive answers directly from the retrieved chunk text. `ANSWER_PROVIDER=openai` remains optional for later use with `OPENAI_ANSWER_MODEL`, but the prompt still instructs the model to answer only from retrieved context.
- `BACKEND_BASE_URL` controls which FastAPI instance the Streamlit frontend talks to; the default local value is `http://127.0.0.1:8000`.
- `VECTORSTORE_DIR` now supports environment-based overrides, which is important for production hosts that require a writable mounted path.
- On Render, the FAISS store should live on the backend service's persistent disk mount such as `/var/data/vectorstore`; otherwise uploads will be lost on restart or redeploy.
- Render documents that free web services have an ephemeral filesystem and that only paid services can attach persistent disks, so this deployment is shareable but not fully no-cost if you want uploads to persist reliably.
- I did not complete a live public deployment from this environment because there is no connected Git provider or confirmed cloud account access here. The repo is prepared for deployment, but you still need to create the Render Blueprint from your own account to obtain public URLs.
