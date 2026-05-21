# Knowledge Assistant Agent

An enterprise-grade AI-powered knowledge assistant that lets you upload documents (PDF, DOCX, CSV, TXT), index them into Pinecone, and query them through a Streamlit chat UI or a REST API.

---

## How It Works

```
User uploads document
        ↓
Document is chunked (800 chars, 100 overlap)
        ↓
Chunks are embedded via OpenAI (text-embedding-3-small)
        ↓
Vectors are stored in Pinecone
        ↓
User asks a question
        ↓
Question is embedded → Pinecone returns top-5 relevant chunks
        ↓
LangGraph agent generates answer using GPT (context-only)
        ↓
Answer + confidence score + sources returned to user
```

---

## Project Structure

```
knowledge-agent/
├── app.py                  # Streamlit frontend (chat UI + file upload)
├── api.py                  # FastAPI backend (REST endpoints)
├── agent.py                # LangGraph agent (retrieve → generate)
├── rag.py                  # Retrieval logic (embed query → query Pinecone)
├── ingest.py               # Document ingestion (load → chunk → embed → upsert)
├── pinecone_client.py      # Pinecone index setup and connection
├── answer_confidence.py    # Confidence scoring (token/phrase overlap)
├── prompts.py              # System prompt for the LLM
├── utils.py                # Indexed docs tracker (indexed_docs.json)
├── uploads/                # Uploaded files are saved here temporarily
├── indexed_docs.json       # Tracks which files have been indexed (local only)
├── Dockerfile              # Docker config for FastAPI (port 8000)
├── Dockerfile.streamlit    # Docker config for Streamlit (port 8501)
├── .streamlit/
│   └── config.toml         # Streamlit server config
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (not committed)
```

---

## Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Pinecone account](https://www.pinecone.io/) with an index created (dimension: `1536`, metric: `cosine`)

---

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd knowledge-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

> The Pinecone index will be auto-created if it doesn't exist when you first run the app.

---

## Running the App

### Option A — Streamlit UI (recommended for local use)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

- Use the **sidebar** to upload documents (PDF, DOCX, CSV, TXT)
- Click **"Upload to Pinecone"** to index the document
- Type questions in the chat input to query your knowledge base

### Option B — FastAPI only

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API Endpoints

| Method | Endpoint  | Description                        |
|--------|-----------|------------------------------------|
| GET    | `/`       | API info                           |
| GET    | `/health` | Health check                       |
| POST   | `/ask`    | Ask a question                     |
| POST   | `/upload` | Upload and index a document        |

### POST `/ask`

```json
// Request
{
  "question": "What are the best practices for Java?",
  "user_id": "user123",       // optional
  "platform": "web"           // optional: teams, slack, whatsapp, web, mobile, api, test
}

// Response
{
  "answer": "...",
  "status": "success",
  "confidence_score": 0.82,
  "confidence_category": "HIGH",
  "is_from_documents": true,
  "sources": [
    { "document": "Effective Java.pdf", "page": 12, "relevance_score": 0.91 }
  ],
  "user_id": "user123"
}
```

### POST `/upload`

Send a `multipart/form-data` request with a `file` field.

```bash
curl -X POST https://<your-render-url>/upload \
  -F "file=@/path/to/document.pdf"
```

```json
// Response
{
  "status": "success",
  "filename": "document.pdf",
  "chunks_indexed": 47
}
```

---

## Running with Docker

### FastAPI

```bash
docker build -f Dockerfile -t knowledge-api .
docker run -p 8000:8000 --env-file .env knowledge-api
```

### Streamlit

```bash
docker build -f Dockerfile.streamlit -t knowledge-streamlit .
docker run -p 8501:8501 --env-file .env knowledge-streamlit
```

---

## Deploying to Render

Both services can be deployed separately on [Render](https://render.com).

### FastAPI Service

| Setting       | Value                                      |
|---------------|--------------------------------------------|
| Runtime       | Docker                                     |
| Dockerfile    | `Dockerfile`                               |
| Port          | `8000`                                     |
| Health Check  | `/health`                                  |

### Streamlit Service

| Setting       | Value                                      |
|---------------|--------------------------------------------|
| Runtime       | Docker                                     |
| Dockerfile    | `Dockerfile.streamlit`                     |
| Port          | `8501`                                     |

### Environment Variables on Render

Add these in the Render dashboard under **Environment**:

```
OPENAI_API_KEY
OPENAI_MODEL
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_CLOUD
PINECONE_REGION
```

> **Note:** The `uploads/` folder and `indexed_docs.json` are ephemeral on Render — they reset on redeploy. All indexed vectors persist in Pinecone regardless.

---

## Confidence Scoring

Every answer includes a confidence score calculated from:

| Component         | Weight | Description                                      |
|-------------------|--------|--------------------------------------------------|
| Best chunk overlap | 40%   | Token overlap between answer and best chunk      |
| Global overlap    | 25%    | Token overlap across all retrieved chunks        |
| Phrase overlap    | 20%    | Exact phrase matches between answer and chunks   |
| Retrieval score   | 15%    | Cosine similarity score from Pinecone            |

| Score     | Category | Meaning                                  |
|-----------|----------|------------------------------------------|
| ≥ 0.8     | HIGH     | Strongly grounded in indexed documents   |
| 0.6–0.79  | MEDIUM   | Reasonably supported                     |
| < 0.6     | LOW      | Weakly grounded, may use general knowledge |

---

## Supported File Types

| Format | Extension |
|--------|-----------|
| PDF    | `.pdf`    |
| Word   | `.docx`   |
| CSV    | `.csv`    |
| Text   | `.txt`    |

---

## Dependencies

| Package              | Purpose                          |
|----------------------|----------------------------------|
| `streamlit`          | Frontend chat UI                 |
| `fastapi` + `uvicorn`| REST API server                  |
| `langgraph`          | Agent orchestration              |
| `langchain-openai`   | LLM integration                  |
| `openai`             | Embeddings + chat completions    |
| `pinecone`           | Vector database                  |
| `PyMuPDF` + `pypdf`  | PDF parsing                      |
| `python-docx`        | Word document parsing            |
| `python-dotenv`      | Environment variable management  |
