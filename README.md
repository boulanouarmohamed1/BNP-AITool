# BNP-Chat


Streamlit workspace for reviewing extracted financial statements, running deterministic and AI-assisted consistency checks, and exporting certified JSON.

## Run with Docker

1. Copy the environment template and replace the Docker Hub image name when you are ready to publish:

   ```sh
   cp .env.example .env
   ```

2. Start the application and Ollama:

   ```sh
   docker compose up --build
   ```

3. Open <http://localhost:8501>.

The first start downloads the configured Ollama model and the RAG embedding model, so it can take several minutes. Ollama models and Hugging Face downloads are retained in Docker volumes for later starts.

Place review inputs in `documents/`. Generated certifications, analysis logs, and exports stay in `certified/`, `logs/`, and `output/`; these folders are deliberately excluded from Git and the Docker image because they can contain client data.


## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint when running outside Compose. |
| `SEMANTIC_VERIFIER_MODEL` | `qwen2.5:3b` | Ollama model for qualitative checks. |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence-transformer model used for PDF retrieval. |
