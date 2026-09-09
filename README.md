# Neri – AI Application with RAG
### About the Project

Neri is a Retrieval-Augmented Generation (RAG) application that allows users to upload documents, search their content, and ask questions through a chat interface.

The system retrieves relevant document chunks before generating an answer, ensuring that responses are grounded in the available knowledge base. It also provides citations so users can verify the source of an answer.

## Problem Statement

Traditional AI assistants can generate plausible answers that are not supported by the provided documents. This makes it difficult for users to verify whether an answer is accurate and based on reliable information.

There is a need for a system that can:

- Retrieve relevant information from documents.
- Generate answers using only the retrieved context.
- Provide citations for generated answers.
- Refuse to answer when sufficient context is unavailable.
- Allow new documents to be added to the knowledge base at runtime.

## Solution

Neri uses a Retrieval-Augmented Generation pipeline to connect document retrieval with answer generation.

### The application:

- Accepts documents through an upload endpoint.
- Ingests and cleans the document content.
- Splits documents into smaller chunks.
- Generates embeddings for the chunks.
- Stores the embeddings in a vector store.
- Retrieves relevant chunks when a user asks a question.
- Generates a grounded answer using the retrieved context.
- Adds citations that map answers back to source chunks.
- Uses hallucination guardrails when retrieval is weak or unavailable.
- Provides the functionality through a backend API and user interface.


## Project Folder Structure

S86-Ai_application_with_RAG-Neri/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── document_intake.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── indexing.py
│   ├── vector_store.py
│   ├── grounded_generation.py
│   ├── citations.py
│   └── guardrails.py
│
├── data/
│   └── documents/
│
├── uploads/
│
├── outputs/
│
├── frontend/
│   └── ...
│
├── requirements.txt
├── .env.example
├── .gitignore
├── api_sample.md
├── upload_sample.md
└── README.md


## How to Run the App
#### 1. Clone the Repository
- git clone https://github.com/kalviumcommunity/S86-Ai_application_with_RAG-Neri.git
- cd S86-Ai_application_with_RAG-Neri

#### 2. Create a Virtual Environment
- python -m venv .venv

#### 3. Activate the Virtual Environment
**Windows**
- .venv\Scripts\activate

#### 4. Install Dependencies
- pip install -r requirements.txt

#### 5. Configure Environment Variables

Create a .env file in the project root.

- OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
- OPENAI_API_KEY=your_api_key_here
- CHAT_MODEL=gemini-3.1-flash-lite
- EMBED_MODEL=gemini-embedding-001

**Note:** Do not commit the actual .env file or API key to GitHub.

#### 6. Start the Backend API
- python -m uvicorn api.main:app --reload

The backend will run at:

- http://127.0.0.1:8000

#### 7. Open the API Documentation

Open the following URL in your browser:

- http://127.0.0.1:8000/docs

FastAPI provides an interactive interface where you can test the available API endpoints.

### Main Features
#### Document Upload

- Users can upload supported documents through the API. Uploaded documents are processed and added to the knowledge base.

#### Document Ingestion

- Documents are loaded and cleaned before being passed to the chunking stage.

#### Chunking

- Large documents are divided into smaller chunks so that relevant sections can be retrieved efficiently.

#### Embedding Generation

- Document chunks are converted into vector embeddings for semantic search.

#### Vector Retrieval

- When a user asks a question, the system retrieves the most relevant document chunks from the knowledge base.

#### Grounded Answer Generation

- The system generates answers using the retrieved context rather than relying only on the model's general knowledge.

#### Source Citations

Generated answers include citation markers such as:

- [1]
- [2]

Each citation can be mapped to information such as the source document, chunk ID, section, and original retrieved text.

#### Hallucination Guardrails

- The system checks the quality of retrieved context before generating an answer.

When sufficient supporting context is not available, the system returns a safe fallback such as:

- I don't have enough reliable context to answer that question.

#### Backend API

The RAG functionality is exposed through backend API endpoints so that a frontend or another application can communicate with the system.

#### Chat / Query Interface

Users can enter questions and receive grounded answers together with the sources used to generate the answer.

#### Streaming Responses

The application supports progressively displaying generated responses instead of waiting for the complete answer.

#### Caching

Repeated identical queries can be served from cache to reduce unnecessary retrieval and generation work.

#### Logging and Usage Monitoring

The application records useful information for monitoring and debugging, including:

- Questions
- Answer previews
- Retrieved sources
- Cache hits
- Request latency
- Token usage
- Estimated cost
- Errors
- Uses

Neri can be used as a document-based knowledge assistant for:

- Internal company documents
- Technical manuals
- Project documentation
- Policies and procedures
- Educational materials
- Research documents
- Frequently asked questions
- Organizational knowledge bases

The system is particularly useful when answers need to be grounded in specific documents and verified through source citations.