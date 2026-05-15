# API Reference — RAG Pipeline

## Endpoints

### Ingest Document

```
POST /api/ingest
Content-Type: multipart/form-data

Fields:
  - file: Document file (PDF, TXT, MD, HTML)
  - chunk_size: Integer (optional, default 1024)
  - chunk_overlap: Integer (optional, default 256)

Response:
{
  "status": "success",
  "chunks_created": 42,
  "source": "filename.pdf"
}
```

### Query

```
POST /api/query
Content-Type: application/json

Body:
{
  "question": "What is the system architecture?",
  "top_k": 5,
  "stream": false,
  "include_sources": true
}

Response:
{
  "answer": "The system consists of four layers...",
  "sources": [
    {
      "source": "architecture.md",
      "chunk_index": 3,
      "score": 0.892,
      "text_preview": "The ingestion layer handles..."
    }
  ]
}
```

### Stream Query

```
POST /api/query
Content-Type: application/json

Body:
{
  "question": "How does authentication work?",
  "stream": true
}

Response: SSE stream
data: {"token": "Authentication", "done": false}
data: {"token": " is", "done": false}
data: {"token": " handled", "done": false}
data: {"token": "...", "done": true}
```

## Rate Limits

- 100 queries/minute per API key
- 10 ingest requests/minute
- Max document size: 50MB
