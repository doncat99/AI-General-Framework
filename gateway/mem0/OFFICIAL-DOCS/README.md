# Mem0 Official Documentation Archive

This directory contains a comprehensive archive of the official Mem0 documentation, scraped and organized for easy reference and development use.

## 📚 Documentation Structure

### Core Documentation Files

1. **[01-introduction.md](./01-introduction.md)** - Introduction to Mem0
   - What is Mem0 and how it works
   - Stateless vs. Stateful Agents
   - Memory vs. RAG comparison
   - Core capabilities and getting started

2. **[02-core-concepts-memory-types.md](./02-core-concepts-memory-types.md)** - Memory Types
   - Understanding different types of memory in AI applications
   - Short-term vs. Long-term memory
   - How Mem0 implements memory systems

3. **[03-memory-operations-search.md](./03-memory-operations-search.md)** - Search Memory Operations
   - How to retrieve relevant memories
   - Search architecture and flow
   - Platform vs. Open Source examples
   - Tips for better search performance

4. **[04-platform-overview.md](./04-platform-overview.md)** - Platform Overview
   - Mem0 managed service features
   - Why choose the platform version
   - Getting started with the platform

5. **[05-open-source-overview.md](./05-open-source-overview.md)** - Open Source Overview
   - Self-hosted Mem0 capabilities
   - Flexible architecture options
   - Core features and integrations

6. **[06-faqs.md](./06-faqs.md)** - Frequently Asked Questions
   - How Mem0 works internally
   - Key features and use cases
   - Troubleshooting and configuration
   - Best practices

## 🎯 Key Concepts

### What is Mem0?

Mem0 is a memory layer designed for modern AI agents that provides:
- **Persistent Memory**: Recall relevant past interactions
- **User Preferences**: Store important user context and preferences
- **Learning Capability**: Learn from successes and failures
- **Seamless Integration**: Easy integration into existing agent stacks

### Platform vs. Open Source

**Mem0 Platform (Managed Service)**
- 4-line integration
- Sub-50ms response times
- Intuitive dashboard
- Scalable infrastructure
- Sign up: https://mem0.dev/pd

**Mem0 Open Source**
- Complete control over infrastructure
- Multiple vector database options (Pinecone, Qdrant, Weaviate, Chroma, PGVector)
- Graph store integration (Neo4j, Memgraph)
- Framework integrations (LangChain, LlamaIndex, AutoGen, CrewAI)

### Memory Types

- **Working Memory**: Short-term session awareness
- **Factual Memory**: Long-term structured knowledge
- **Episodic Memory**: Records of specific past conversations
- **Semantic Memory**: General knowledge built over time

## 🔧 Technical Architecture

### Search Architecture Flow
1. **Query Processing** - LLM refines natural language queries
2. **Vector Search** - Semantic embeddings with cosine similarity
3. **Filtering & Ranking** - Logical filters and scoring
4. **Results Delivery** - Relevant memories with metadata

### Memory vs. RAG Comparison

| Aspect | RAG | Mem0 Memory |
|--------|-----|-------------|
| Statefulness | Stateless | Stateful |
| Recall Type | Document lookup | Evolving user context |
| Use Case | Ground answers in data | Guide behavior across time |
| Retention | Temporary | Persistent |
| Personalization | None | Deep, evolving profile |

## 🚀 Common Use Cases

- **Personalized Learning Assistants** - Remember user progress and preferences
- **Customer Support AI Agents** - Context-aware assistance across sessions
- **Healthcare Assistants** - Track patient history and treatment plans
- **Virtual Companions** - Build deeper relationships through memory
- **Productivity Tools** - Remember user habits and workflows
- **Gaming AI** - Adaptive experiences based on player history

## 💡 Best Practices

### Memory Creation
- Include temporal markers (when events occurred)
- Add personal context or experiences
- Frame information in real-world applications
- Use specific examples rather than general definitions

### Search Optimization
- Use descriptive natural queries
- Apply filters for scoped, faster lookup
- Use `version: "v2"` for enhanced results
- Consider wildcard filters for broader matches
- Tune with `top_k`, `threshold`, or `rerank` parameters

### Metadata Usage
- Store contextual information (location, time, device)
- Include user attributes (preferences, skill levels)
- Add interaction details (topics, sentiment, urgency)
- Use custom tags for domain-specific categorization

## 🔗 Important Links

- **Official Website**: https://mem0.ai
- **GitHub Repository**: https://github.com/mem0ai/mem0
- **Platform Signup**: https://mem0.dev/pd
- **Discord Community**: https://mem0.dev/DiD
- **API Documentation**: https://docs.mem0.ai/api-reference
- **Examples**: https://docs.mem0.ai/examples
- **Integrations**: https://docs.mem0.ai/integrations

## 📊 Site Structure Overview

Based on the comprehensive site mapping, the Mem0 documentation includes:

### Main Sections
- **Getting Started** (Introduction, Quickstart, FAQs)
- **Core Concepts** (Memory Types, Operations)
- **Platform** (Overview, Features, Quickstart)
- **Open Source** (Overview, Features, Components)
- **Examples** (Real-world implementations)
- **Integrations** (Framework connections)
- **API Reference** (Complete API documentation)
- **Components** (Vector DBs, LLMs, Embedders)

### Total Pages Discovered
86+ documentation pages covering all aspects of Mem0 implementation and usage.

## 🛠️ Configuration Examples

### AWS Lambda Configuration
```bash
MEM0_DIR=/tmp/.mem0
```

### Disable Telemetry
```bash
MEM0_TELEMETRY=False
```

### Platform Usage
```python
from mem0 import MemoryClient

client = MemoryClient(api_key="your-api-key")
results = client.search(query, version="v2", filters=filters)
```

### Open Source Usage
```python
from mem0 import Memory

m = Memory()
related_memories = m.search("query", user_id="alice")
```

## 📝 Archive Information

- **Archive Date**: January 9, 2025
- **Source**: https://docs.mem0.ai/
- **Scraping Method**: Firecrawl MCP Server
- **Total Files**: 6 core documentation files
- **Coverage**: Comprehensive overview of key concepts and features

This archive provides a solid foundation for understanding and implementing Mem0 in your AI applications. For the most up-to-date information, always refer to the official documentation at https://docs.mem0.ai/.
## Project Integration Notes

In this project, the Mem0 service is bundled as a FastAPI server that exposes:

- `GET /health` – health check
- `POST /add` – add memory: `{ message, user_id?, agent_id?, metadata? }`
- `POST /search` – search memory: `{ query, user_id?, agent_id?, limit? }`
- `GET /memories?user_id=...` – retrieve stored memories

Embeddings are computed locally via Ollama to avoid external dependency or cost during demos:

- Container env: `OLLAMA_HOST=http://host.docker.internal:11434`
- Config: `mem0/config.py` sets embedder provider to `ollama`

Quick check (PowerShell):

```
$add = @{ message='remember that user prefers detailed technical analysis'; user_id='verification_test' } | ConvertTo-Json
Invoke-WebRequest -Method Post -ContentType 'application/json' -Uri 'http://localhost:8001/add' -Body $add | Select-Object -Expand Content
Invoke-RestMethod 'http://localhost:8001/memories?user_id=verification_test' | ConvertTo-Json -Depth 10
```
