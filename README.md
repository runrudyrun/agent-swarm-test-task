# InfinitePay Agent Swarm 🤖

An intelligent multi-agent system for customer support and knowledge retrieval, developed for InfinitePay as a test task. The system uses specialized agents to provide accurate and personalized responses to users.

## 🚀 Features

- **Multi-Agent Architecture**: System with three specialized agents:
  - **Router Agent**: Intent classification and intelligent routing
  - **Knowledge Agent**: RAG (Retrieval-Augmented Generation) over InfinitePay content
  - **Support Agent**: Tools for user data and support
- **FastAPI**: Modern and high-performance REST API
- **RAG Pipeline**: Complete knowledge ingestion and retrieval system
- **Adjustable Personality**: Optional personality layer for friendly tone
- **Dockerized**: Complete containerization with health checks
- **Complete Tests**: Unit and E2E test suite
- **Portuguese Language**: Optimized for Brazilian users

## 📋 Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional)
- OpenAI API key (optional - for better quality)

## 🔧 Installation

### Method 1: Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-swarm-test-task
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -e .
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Method 2: Docker

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-swarm-test-task
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run with Docker Compose:
```bash
docker-compose up -d
```

## 🏃‍♂️ Usage

### Start Server

**Local:**
```bash
uvicorn api.main:app --reload
```

**Docker:**
```bash
docker-compose up -d
```

### Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Send a Query:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Qual é o meu saldo?",
    "user_id": "user123"
  }'
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📊 Architecture

### Main Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Router Agent  │───▶│ Knowledge Agent │───▶│   Vector Store  │
│  (Classification)│    │    (RAG)        │    │   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Support Agent  │───▶│   User Store    │    │   Mock Data     │
│  (Tools)        │    │   (JSON Mock)   │    │  (users.json)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Processing Flow

1. **Reception**: Query received via POST /query
2. **Routing**: Router Agent classifies intent
3. **Processing**: Query routed to appropriate agent
4. **Response**: Response processed and personalized
5. **Delivery**: Formatted response sent to user

## 🧪 Tests

### Run All Tests
```bash
pytest
```

### Tests with Coverage
```bash
pytest --cov=. --cov-report=html
```

### Specific Tests
```bash
# Router Agent tests
pytest tests/test_router.py

# Support Agent tests
pytest tests/test_support.py

# Knowledge Agent tests
pytest tests/test_knowledge.py

# E2E tests
pytest tests/test_api_e2e.py
```

## 📚 Configuration

### Environment Variables

| Variable | Description | Default | Options |
|----------|-----------|---------|---------|
| `MODEL_PROVIDER` | LLM provider | `local` | `local`, `openai` |
| `OPENAI_API_KEY` | OpenAI API key | - | string |
| `EMBEDDINGS_PROVIDER` | Embeddings provider | `local` | `local`, `openai` |
| `VECTOR_STORE` | Vector store | `chroma` | `chroma`, `faiss` |
| `LOCALE` | Language/locale | `pt-BR` | `pt-BR` |
| `PERSONALITY` | Personality layer | `on` | `on`, `off` |
| `PORT` | Server port | `8000` | 1-65535 |
| `LOG_LEVEL` | Log level | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Development Settings

**Development with Hot Reload:**
```bash
docker-compose --profile dev up -d
```

**With OpenAI:**
```bash
docker-compose --profile openai up -d
```

## 🔍 Data Ingestion (RAG)

### Run Ingestion
```bash
python -m rag.ingest
```

### Processed URLs
The system automatically collects information from:
- https://www.infinitepay.io
- https://www.infinitepay.io/maquininha
- https://www.infinitepay.io/maquininha-celular
- https://www.infinitepay.io/tap-to-pay
- https://www.infinitepay.io/pdv
- https://www.infinitepay.io/receba-na-hora
- https://www.infinitepay.io/gestao-de-cobranca-2
- https://www.infinitepay.io/gestao-de-cobranca
- https://www.infinitepay.io/link-de-pagamento
- https://www.infinitepay.io/loja-online
- https://www.infinitepay.io/boleto
- https://www.infinitepay.io/conta-digital
- https://www.infinitepay.io/conta-pj
- https://www.infinitepay.io/pix
- https://www.infinitepay.io/pix-parcelado
- https://www.infinitepay.io/emprestimo
- https://www.infinitepay.io/cartao
- https://www.infinitepay.io/rendimento

## 🎯 Usage Examples

### Balance Query (Support Agent)
```json
{
  "message": "Qual é o meu saldo?",
  "user_id": "user123"
}
```

Response:
```json
{
  "answer": "📋 **Account Details**\n\n**Name:** João Silva\n**Current Balance:** R$ 1,250.75\n**Status:** ✅ Active",
  "agent_used": "support",
  "intent": "support",
  "confidence": 0.9
}
```

### Product Information (Knowledge Agent)
```json
{
  "message": "Como funciona a maquininha?"
}
```

Response:
```json
{
  "answer": "The InfinitePay card machine allows you to accept card payments...",
  "agent_used": "knowledge",
  "intent": "knowledge",
  "confidence": 0.85,
  "sources": ["https://www.infinitepay.io/maquininha"]
}
```

### Multi-Intent
```json
{
  "message": "Quero saber meu saldo e também como funciona a maquininha",
  "user_id": "user123"
}
```

## 🔧 Development

### Project Structure

```
agent-swarm-test-task/
├── api/                    # FastAPI application
│   ├── main.py            # Main application
│   └── schemas.py         # Pydantic models
├── agents/                 # Agent implementations
│   ├── router_agent.py    # Intent classification
│   ├── knowledge_agent.py # RAG implementation
│   ├── support_agent.py   # Support tools
│   └── personality.py     # Tone adjustment
├── rag/                    # RAG pipeline
│   ├── config.py          # RAG configuration
│   └── ingest.py          # Data ingestion
├── tools/                  # Agent tools
│   ├── user_store.py      # Mock user data
│   └── web_search.py      # Web search (optional)
├── tests/                  # Test suite
├── data/                   # Data storage
│   ├── raw/               # Raw scraped content
│   ├── chroma/            # Vector database
│   ├── mock/              # Mock user data
│   └── sources/           # URL sources tracking
├── pyproject.toml         # Python dependencies
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-service setup
└── README.md             # This file
```

### Adding New Agents

1. Create a new file in `agents/`
2. Implement the agent class following the existing pattern
3. Add to the Router Agent
4. Create tests in `tests/`
5. Update documentation

### Customization

**Adjust Personality:**
Edit `agents/personality.py` to modify response tone.

**Add New URLs:**
Add to the `INFINITEPAY_URLS` list in `rag/config.py`.

**Modify Routing Rules:**
Edit patterns in `agents/router_agent.py`.

## 🚨 Troubleshooting

### Common Issues

**1. Vector Store won't load:**
- Check if `data/chroma` directory exists
- Run ingestion: `python -m rag.ingest`
- Check file permissions

**2. Agents not responding:**
- Check logs: `docker-compose logs agent-swarm`
- Test health check: `curl http://localhost:8000/health`
- Check environment variables

**3. Tests failing:**
- Check dependencies: `pip install -e .[dev]`
- Make sure test environment is clean
- Run with debug: `pytest -v -s`

### Debug

**Enable Debug Logging:**
```bash
export LOG_LEVEL=DEBUG
```

**Check Settings:**
```bash
curl http://localhost:8000/capabilities
```

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


---

**Developed with ❤️ for InfinitePay**