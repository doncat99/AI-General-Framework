# Search Memory - Mem0 Memory Operations

## Overview

The `search` operation allows you to retrieve relevant memories based on a natural language query and optional filters like user ID, agent ID, categories, and more. This is the foundation of giving your agents memory-aware behavior.

Mem0 supports:

- Semantic similarity search
- Metadata filtering (with advanced logic)
- Reranking and thresholds
- Cross-agent, multi-session context resolution

This applies to both:

- **Mem0 Platform** (hosted API with full-scale features)
- **Mem0 Open Source** (local-first with LLM inference and local vector DB)

## Architecture

![Search Architecture](https://mintcdn.com/mem0/k4LG2Flv5533NbwL/images/search_architecture.png)

Architecture diagram illustrating the memory search process.

The search flow follows these steps:

1. **Query Processing**
   An LLM refines and optimizes your natural language query.

2. **Vector Search**
   Semantic embeddings are used to find the most relevant memories using cosine similarity.

3. **Filtering & Ranking**
   Logical and comparison-based filters are applied. Memories are scored, filtered, and optionally reranked.

4. **Results Delivery**
   Relevant memories are returned with associated metadata and timestamps.

## Example: Mem0 Platform

```python
from mem0 import MemoryClient

client = MemoryClient(api_key="your-api-key")

query = "What do you know about me?"
filters = {
   "OR": [
      {"user_id": "alice"},
      {"agent_id": {"in": ["travel-assistant", "customer-support"]}}
   ]
}

results = client.search(query, version="v2", filters=filters)
```

## Example: Mem0 Open Source

```python
from mem0 import Memory

m = Memory()
related_memories = m.search("Should I drink coffee or tea?", user_id="alice")
```

## Tips for Better Search

- Use descriptive natural queries (Mem0 can interpret intent)
- Apply filters for scoped, faster lookup
- Use `version: "v2"` for enhanced results
- Consider wildcard filters (e.g., `run_id: "*"`) for broader matches
- Tune with `top_k`, `threshold`, or `rerank` if needed

### More Details

For the full list of filter logic, comparison operators, and optional search parameters, see the [Search Memory API Reference](https://docs.mem0.ai/api-reference/memory/v2-search-memories).

## Need help?

If you have any questions, please feel free to reach out to us using one of the following methods:

- **Discord**: Join our community at https://mem0.dev/DiD
- **GitHub**: Ask questions on GitHub at https://github.com/mem0ai/mem0/discussions/new?category=q-a
- **Support**: Talk to founders at https://cal.com/taranjeetio/meet
