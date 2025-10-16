# Minimal async FastAPI + PostgreSQL template  
**with Google ADK agent, Langfuse tracing, mem0 memory, and RAG ingestion**

[![Live example](https://img.shields.io/badge/live%20example-https%3A%2F%2Fminimal--fastapi--postgres--template.rafsaf.pl-blueviolet)](https://minimal-fastapi-postgres-template.rafsaf.pl/)
[![License](https://img.shields.io/github/license/rafsaf/minimal-fastapi-postgres-template)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://docs.python.org/3/whatsnew/3.13.html)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/rafsaf/minimal-fastapi-postgres-template/actions/workflows/tests.yml/badge.svg)](https://github.com/rafsaf/minimal-fastapi-postgres-template/actions/workflows/tests.yml)

> This template adds an **AI assistant** powered by **Google ADK**, **OpenRouter/LiteLLM**, **Langfuse** tracing, **mem0** memory, and a **CLI** to ingest content (mem0 / Chroma / Neo4j) and chat.

---

## Features

- ✅ **FastAPI + PostgreSQL 16** (async SQLAlchemy 2.0, Alembic, Poetry, Ruff, mypy)
- ✅ **Auth & tests**: refresh token flow, +40 async tests, pytest-xdist, coverage
- ✅ **Google ADK Agent**
  - `utilities/base/base_agent.py` (ADK + LiteLLM bridge via OpenRouter)
  - `agents/assistant/adk_client.py` (thin client subclass with dynamic tools)
- ✅ **Tracing** with **Langfuse** via OpenInference
- ✅ **Long-term memory** with **mem0**
- ✅ **RAG ingestion** framework:
  - Sources: **JSON** (roadmap), **PDF (PyMuPDF)**
  - Sinks: **mem0**, **Chroma** (vector), **Neo4j** (graph)
  - Local embeddings via **Sentence-Transformers**
- ✅ **CLI assistant**: ingest + single-turn ask + multi-turn chat + tool listing
- ✅ **Tooling model**: builtin tools & external plugin loader

---

## Quickstart

### 1) Install dependencies

```bash
# Python 3.13 recommended
poetry install
# or use your own venv + pip
```

**AI / RAG extras:**
```bash
pip install   google-adk vertexai-agent-engines google-genai   litellm openinference-instrumentation-google-adk   langfuse mem0ai   chromadb sentence-transformers torch   pymupdf neo4j   pydantic-settings loguru tqdm
```

> **Note:** Pydantic v2 moved `BaseSettings` to **pydantic-settings**.

### 2) Configure environment

Create `.env` (or export in shell):

```bash
# OpenRouter (LiteLLM backend)
OPEN_ROUTER_API_KEY="sk-or-..."
OPEN_ROUTER_API_BASE="https://openrouter.ai/api/v1"   # optional if default

# Model used by ADK (via OpenRouter)
ADK_MODEL="google/gemini-1.5-flash"                   # default if unset

# Langfuse (optional but recommended)
LANGFUSE_HOST="https://cloud.langfuse.com"
LANGFUSE_PUBLIC_KEY="pk_..."
LANGFUSE_SECRET_KEY="sk_..."

# mem0
MEM0_API_KEY="..."

# Neo4j (if using graph sink)
NEO4J_URL="neo4j+s://<your-aura-uri>"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="****"
```

### 3) Database & migrations

```bash
docker-compose up -d
alembic upgrade head
```

### 4) Run the FastAPI app

```bash
uvicorn app.main:app --reload
# Open http://localhost:8000/  (OpenAPI docs at "/")
```

---

## AI Assistant Architecture

**Where things live:**

- **Agent & runtime**
  - `utilities/base/base_agent.py` — builds a **Google ADK** `LlmAgent` using **LiteLLM** (OpenRouter) and instruments traces with **OpenInference** (Langfuse).
  - `agents/assistant/adk_client.py` — subclass of `BaseAgent` with:
    - `register_tool()` / `register_tools()` for dynamic tool registration
    - `run(prompt, system_prompt=...)` for simple single-turn asks
- **Tools**
  - Builtins: `agents/assistant/tools/builtin_tools.py`
  - External plugin loader: `agents/assistant/tools/external_loader.py`
- **Memory**
  - **mem0** used as a sink in ingestion; can be wired at runtime too

---

## Ingestion Framework (RAG)

- **Entry:** `run_assistant.py` (unified CLI)
- **Sources:**
  - JSON roadmap → `agents/assistant/ingest/sources/json_roadmap.py`
  - PDF (PyMuPDF) → `agents/assistant/ingest/sources/pdf_pymupdf.py`
- **Sinks:**
  - mem0 → `agents/assistant/ingest/sinks/mem0_sink.py`
  - Chroma → `agents/assistant/ingest/sinks/chroma_sink.py`
  - Neo4j → `agents/assistant/ingest/sinks/neo4j_sink.py`
- **Pipeline:** `agents/assistant/ingest/pipeline.py`

**No LangChain required** for ingestion/retrieval. We use `chromadb` directly and **Sentence-Transformers** for embeddings; PDFs via **PyMuPDF**.

---

## CLI: ingest & chat

Launch the CLI:

```bash
python run_assistant.py
```

You’ll see:

```
========== Assistant ==========
  1) Ingest
  2) Ask (single turn)
  3) Chat (multi-turn)
  4) List tools
  0) Exit
Choose an option:
```

### 1) Ingest
Pick a source (JSON/PDF) and one or more sinks (mem0 / Chroma / Neo4j).  
Example:
```
Source file path: /path/to/backend_developer_roadmap_gfg.json
Sinks (comma separated): mem0,chroma
```

### 2) Ask (single turn)
One-off prompt to the **ADK agent** (uses registered tools; emits traces to Langfuse if configured).

### 3) Chat (multi-turn)
Maintains an **ADK** in-memory session for back-and-forth.

### 4) List tools
Shows all registered tools (builtin + external plugins).

---

## Adding Tools

Create plain Python callables with type hints and a docstring:

```python
# agents/assistant/tools/agent_tools.py
from typing import Dict, Any, List

def get_time(city: str) -> Dict[str, Any]:
    """Return the local time for a city."""
    ...

def get_agent_tools() -> List[callable]:
    return [get_time]
```

They’ll be auto-loaded by `external_loader` if present; or register at runtime:

```python
from agents.assistant.adk_client import ADKClient
from agents.assistant.tools.builtin_tools import builtin_tools

client = ADKClient()
# builtins
for _, fn in builtin_tools().items():
    client.register_tool(fn)

# external pack
from agents.assistant.tools.agent_tools import get_agent_tools
client.register_tools(get_agent_tools())
```

---

## Troubleshooting

- **Pydantic v2 `BaseSettings` error**  
  Install `pydantic-settings` and update imports:
  ```python
  from pydantic_settings import BaseSettings
  ```

- **Sentence-Transformers / Torch**  
  Ensure `torch` installs for your platform. For CPU:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install sentence-transformers
  ```

- **OpenRouter model IDs**  
  We normalize as `openrouter/<model>`. Set:
  ```
  ADK_MODEL=google/gemini-1.5-flash
  ```
  The bridge will use `openrouter/google/gemini-1.5-flash`.

- **Langfuse traces not visible**  
  Verify `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and ensure `OPENINFERENCE_DISABLED` is **not** `1`.

---

## Step-by-step FastAPI example (from the base template)

The original tutorial (Pets CRUD, migrations, tests) still applies:
- Create SQLAlchemy model
- Create & apply Alembic migration
- Create request/response schemas
- Create endpoints
- Write tests

*(See the upstream template docs for full details.)*

---

## Design Notes

- **Deployment (Docker):** The included Dockerfile runs FastAPI with uvicorn. The AI assistant/ingestion runs as a CLI; you can containerize it or expose via an API route if desired.
- **CORS, Allowed Hosts, Docs URL:** Same as the template (see `app/main.py`), adjust for production.

---

## License

MIT — see [LICENSE](LICENSE).

---

### What’s new vs. the original template?

- **Added:** Google **ADK** agent (`BaseAgent`, `ADKClient`), **Langfuse** tracing
- **Added:** **mem0** long-term memory
- **Added:** **RAG ingestion** (PyMuPDF, Chroma, Neo4j, Sentence-Transformers)
- **Added:** **CLI assistant** (ingest + ask + chat + tools)
- **Updated:** Config to **pydantic-settings** (Pydantic v2)

Happy hacking!
