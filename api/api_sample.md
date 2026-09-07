# RAG API Sample Request & Response

## Endpoint

**Method:** `POST`

**URL:** `http://127.0.0.1:8000/query`

---

## Request

### Headers

```text
Content-Type: application/json
```

### Body

```json
{
  "question": "What should a technician do if abnormal vibration is detected?"
}
```

---

## Response

### Status Code

```text
200 OK
```

### Body

```json
{
  "answer": "If abnormal vibration is detected, the technician should stop the machine immediately and begin the approved inspection procedure. The equipment should not be restarted until the inspection is completed and the machine is considered safe.",
  "sources": [
    {
      "source": "vibration_manual.txt",
      "chunk_id": "vibration_manual.txt:0",
      "score": 0.6435657441616058
    },
    {
      "source": "vibration_procedure.txt",
      "chunk_id": "vibration_procedure.txt:0",
      "score": 0.6008005142211914
    },
    {
      "source": "machine_manual.txt",
      "chunk_id": "machine_manual.txt:2",
      "score": 0.5966055989265442
    }
  ],
  "status": "answered"
}
```

---

## cURL Request

### Windows

```cmd
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What should a technician do if abnormal vibration is detected?\"}"
```

---

## Input Validation

The API validates the `question` field before running the RAG pipeline.

### Missing Question

Request:

```json
{}
```

Response:

```text
422 Unprocessable Entity
```

### Question Too Short

Request:

```json
{
  "question": "Hi"
}
```

Response:

```text
422 Unprocessable Entity
```

### Question Too Long

Questions exceeding the configured maximum length are rejected.

Response:

```text
422 Unprocessable Entity
```

---

## Missing Context / Refusal

When the retrieved context is not strong enough, the API returns a safe refusal instead of generating an unsupported answer.

### Request

```json
{
  "question": "What is the company's refund policy?"
}
```

### Response

```json
{
  "answer": "I don't have enough reliable context to answer that question.",
  "sources": [],
  "status": "refused"
}
```

---

## API Documentation

When the server is running, the interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The API provides a structured JSON interface for sending questions to the RAG pipeline and receiving grounded answers with supporting sources.