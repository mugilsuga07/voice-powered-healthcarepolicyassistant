# Voice-Powered Healthcare Policy Assistant

A voice-enabled AI assistant that helps users query healthcare insurance policy documents using natural language. Built with Retrieval-Augmented Generation (RAG) for accurate, grounded responses.

## Features

- **Voice Input**: Record questions using your microphone
- **Text Input**: Type questions directly
- **RAG-based Retrieval**: Searches policy documents using vector similarity
- **Voice Output**: Converts answers to speech using text-to-speech
- **Plan Filtering**: Filter search results by specific insurance plans
- **Source Citations**: Shows relevant policy excerpts with page references
- **PII Sanitization**: Automatically redacts sensitive information from queries

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Voice Input   │────▶│  Speech-to-Text │────▶│   Sanitizer     │
│   (Microphone)  │     │  (Whisper API)  │     │  (PII Redaction)│
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│   Text Input    │────▶│   Sanitizer     │────▶│   RAG Pipeline  │
│   (Keyboard)    │     │  (PII Redaction)│     │  (ChromaDB +    │
└─────────────────┘     └─────────────────┘     │   GPT-4o-mini)  │
                                                └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│  Audio Player   │◀────│  Text-to-Speech │◀────│  Answer + Cites │
│                 │     │  (OpenAI TTS)   │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Speech-to-Text | OpenAI Whisper API |
| Text-to-Speech | OpenAI TTS API |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | ChromaDB |
| LLM | GPT-4o-mini |
| PDF Processing | PyMuPDF |

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd voice-powered-healthcare-insurancepolicy-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Add policy documents**
   Place PDF files in `data/policies/`

6. **Ingest documents**
   ```bash
   python scripts/ingest_documents.py
   ```

7. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
├── app.py                    # Main Streamlit application
├── services/
│   ├── stt.py               # Speech-to-text service
│   ├── tts.py               # Text-to-speech service
│   ├── rag.py               # RAG pipeline (retrieval + generation)
│   └── sanitizer.py         # PII detection and redaction
├── scripts/
│   └── ingest_documents.py  # Document ingestion script
├── data/
│   ├── policies/            # PDF policy documents
│   └── chroma/              # Vector database storage
├── requirements.txt
└── README.md
```

## Usage

1. Open the application at `http://localhost:8501`
2. Select an insurance plan from the sidebar (or search all plans)
3. Ask a question using voice or text input
4. View the answer with source citations
5. Listen to the audio response

## Sample Queries

- "What is the deductible?"
- "What is the out-of-pocket maximum?"
- "Does this plan cover emergency room visits?"
- "What is the copay for a specialist visit?"
- "Is preventive care covered?"

## Key Implementation Details

### RAG Pipeline
- Documents are chunked using recursive character splitting (500 chars, 100 overlap)
- Embeddings are generated using OpenAI's text-embedding-3-small model
- Retrieval uses L2 distance with similarity threshold filtering
- Top 5 relevant chunks are passed to GPT-4o-mini for answer generation

### Guardrails
- Medical advice questions are automatically blocked
- Responses are grounded only in retrieved policy documents
- PII is redacted before processing

### Query Preprocessing
- Conversational filler is stripped for better retrieval accuracy
- Common patterns like "Hey, my name is..." are removed

## Requirements

- Python 3.10+
- OpenAI API key
- Microphone access (for voice input)

## License

MIT License
