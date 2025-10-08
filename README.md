# AIQ Clean Architecture Implementation

A simple LangChain LLM | Parser implementation following Clean Code Architecture principles, fully integrated with NVIDIA's AIQ/NeMo Agent Toolkit.

## Architecture

This project follows Clean Architecture with clear separation of concerns:

```
src/
├── domain/              # Enterprise Business Rules
│   ├── entities/       # Domain entities (ChatMessage, ChatResponse)
│   └── interfaces/     # Abstract interfaces (ILLMService, IOutputParser)
├── application/        # Application Business Rules
│   └── use_cases/     # Use cases (ChatUseCase, StreamChatUseCase)
├── infrastructure/     # Frameworks & Drivers
│   ├── llm/           # LLM adapters (LangChain)
│   └── parsers/       # Parser implementations (JSON, LangChain)
└── presentation/       # Interface Adapters
    └── api/           # FastAPI routes and schemas
```

### Clean Architecture Principles

- **Dependency Inversion**: Domain layer defines interfaces; infrastructure implements them
- **Separation of Concerns**: Each layer has a single responsibility
- **Framework Independence**: Domain logic is independent of FastAPI, LangChain, etc.
- **Testability**: Easy to mock dependencies and test each layer independently

## Features

- ✅ Clean Architecture structure
- ✅ **AIQ/NeMo Agent Toolkit** integration
- ✅ LangChain LLM integration (OpenAI, NVIDIA NIM)
- ✅ Output parsing (JSON, extensible to LangChain parsers)
- ✅ REST API with `/chat` endpoint
- ✅ Streaming support with `/chat/stream` endpoint
- ✅ Dependency injection
- ✅ Environment-based configuration
- ✅ **uv** package manager

## Prerequisites

- Python 3.11 or 3.12 (AIQ toolkit requires Python >=3.11, <3.13)
- [uv](https://docs.astral.sh/uv/) package manager

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

1. Clone the repository
2. Create a venv 
```bash
uv venv
```
3. Activate the venv
```bash
source .venv/bin/activate
```
3. Install dependencies with uv:
```bash
uv pip install -e .
```

For development dependencies:
```bash
uv pip install -e ".[dev]"
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

### Understanding the Architecture

This implementation uses:
- **AIQ Toolkit** - NVIDIA's agent orchestration framework (runs locally, no API key needed)
- **LLM Provider** - The actual AI model service (requires API key)

### LLM Provider Setup

You need to choose an LLM provider and configure its API key:

#### Option 1: OpenAI (Recommended for getting started)

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_BASE_URL=  # Leave empty for standard OpenAI API
```

Get your OpenAI API key from: https://platform.openai.com/api-keys

#### Option 2: NVIDIA NIM (NVIDIA's LLM Service)

```env
LLM_PROVIDER=nvidia
LLM_MODEL=meta/llama-3.1-8b-instruct  # or other NVIDIA models
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

Get your NVIDIA API key from: https://build.nvidia.com/

**Note**: The AIQ Toolkit itself doesn't require an API key - it's a local orchestration framework. You only need API keys for the LLM providers (OpenAI or NVIDIA NIM) that actually run the AI models.

## Usage

### Start the API server

```bash
python -m src.main
```

Or with uvicorn:
```bash
uvicorn src.main:app --reload
```

### API Endpoints

#### POST /api/v1/chat

Send a chat message and get a response:

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is clean architecture?"}
    ],
    "parse_output": true
  }'
```

Response:
```json
{
  "message": {
    "role": "assistant",
    "content": "Clean architecture is...",
    "metadata": null
  },
  "parsed_output": {...}
}
```

#### POST /api/v1/chat/stream

Stream a chat response:

```bash
curl -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ]
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Extending the Implementation

### Adding a New LLM Provider

1. Create adapter in `src/infrastructure/llm/`:
```python
class CustomLLMAdapter(ILLMService):
    async def chat(self, messages, system_prompt):
        # Implementation

    async def stream(self, messages, system_prompt):
        # Implementation
```

2. Register in `src/dependencies.py`

### Adding a New Parser

1. Create parser in `src/infrastructure/parsers/`:
```python
class CustomParser(IOutputParser):
    def parse(self, text: str) -> dict:
        # Implementation
```

2. Register in `src/dependencies.py`

## AIQ Compatibility

This implementation is compatible with NVIDIA's AIQ/NeMo Agent Toolkit:

- **Framework Agnostic**: Core domain logic independent of frameworks
- **Modular Design**: Easy to integrate with AIQ agents and workflows
- **Composable**: Use cases can be composed into larger agent systems
- **Extensible**: Plugin architecture for LLMs and parsers
- **Production Ready**: Async support, streaming, error handling

You can use this as a building block in larger AIQ multi-agent systems.

## Dependencies

Key dependencies installed:
- **aiqtoolkit** - NVIDIA NeMo Agent Toolkit (local orchestration, no API key needed)
- **langchain** - LangChain framework
- **langchain-openai** - OpenAI LLM integration (requires OpenAI API key)
- **langchain-nvidia-ai-endpoints** - NVIDIA NIM LLM integration (requires NVIDIA API key)
- **fastapi** - Web framework
- **uvicorn** - ASGI server

See `pyproject.toml` for full dependency list.

### What Needs API Keys?

- ❌ **AIQ Toolkit** - No API key needed (runs locally)
- ❌ **LangChain** - No API key needed (framework only)
- ❌ **FastAPI/Uvicorn** - No API key needed (web server)
- ✅ **OpenAI** - Requires API key if using `LLM_PROVIDER=openai`
- ✅ **NVIDIA NIM** - Requires API key if using `LLM_PROVIDER=nvidia`

## Testing

```bash
uv run pytest tests/
```

## License

MIT
