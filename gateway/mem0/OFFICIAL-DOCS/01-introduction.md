# Introduction - Mem0

Mem0 is a memory layer designed for modern AI agents. It acts as a persistent memory layer that agents can use to:

- Recall relevant past interactions
- Store important user preferences and factual context
- Learn from successes and failures

It gives AI agents memory so they can remember, learn, and evolve across interactions. Mem0 integrates easily into your agent stack and scales from prototypes to production systems.

## Stateless vs. Stateful Agents

Most current agents are stateless: they process a query, generate a response, and forget everything. Even with huge context windows, everything resets the next session.

Stateful agents, powered by Mem0, are different. They retain context, recall what matters, and behave more intelligently over time.

![Stateless vs Stateful Agents](https://mintlify.s3.us-west-1.amazonaws.com/mem0/images/stateless-vs-stateful-agent.png)

## Where Memory Fits in the Agent Stack

Mem0 sits alongside your retriever, planner, and LLM. Unlike retrieval-based systems (like RAG), Mem0 tracks past interactions, stores long-term knowledge, and evolves the agent's behavior.

![Memory Agent Stack](https://mintlify.s3.us-west-1.amazonaws.com/mem0/images/memory-agent-stack.png)

Memory is not about pushing more tokens into a prompt but about intelligently remembering context that matters. This distinction matters:

| Capability | Context Window | Mem0 Memory |
| --- | --- | --- |
| Retention | Temporary | Persistent |
| Cost | Grows with input size | Optimized (only what matters) |
| Recall | Token proximity | Relevance + intent-based |
| Personalization | None | Deep, evolving profile |
| Behavior | Reactive | Adaptive |

## Memory vs. RAG: Complementary Tools

RAG (Retrieval-Augmented Generation) is great for fetching facts from documents. But it's stateless. It doesn't know who the user is, what they've asked before, or what failed last time.

Mem0 provides continuity. It stores decisions, preferences, and context—not just knowledge.

| Aspect | RAG | Mem0 Memory |
| --- | --- | --- |
| Statefulness | Stateless | Stateful |
| Recall Type | Document lookup | Evolving user context |
| Use Case | Ground answers in data | Guide behavior across time |

Together, they're stronger: RAG informs the LLM; Mem0 shapes its memory.

## Types of Memory in Mem0

Mem0 supports different kinds of memory to mimic how humans store information:

- **Working Memory**: short-term session awareness
- **Factual Memory**: long-term structured knowledge (e.g., preferences, settings)
- **Episodic Memory**: records specific past conversations
- **Semantic Memory**: builds general knowledge over time

## Why Developers Choose Mem0

Mem0 isn't a wrapper around a vector store. It's a full memory engine with:

- **LLM-based extraction**: Intelligently decides what to remember
- **Filtering & decay**: Avoids memory bloat, forgets irrelevant info
- **Costs Reduction**: Save compute costs with smart prompt injection of only relevant memories
- **Dashboards & APIs**: Observability, fine-grained control
- **Cloud and OSS**: Use our platform version or our open-source SDK version

You plug Mem0 into your agent framework, it doesn't replace your LLM or workflows. Instead, it adds a smart memory layer on top.

## Core Capabilities

- **Reduced token usage and faster responses**: sub-50 ms lookups
- **Semantic memory**: procedural, episodic, and factual support
- **Multimodal support**: handle both text and images
- **Graph memory**: connect insights and entities across sessions
- **Host your way**: either a managed service or a self-hosted version

## Getting Started

Mem0 offers two powerful ways to leverage our technology: our [managed platform](https://docs.mem0.ai/platform/overview) and our [open source solution](https://docs.mem0.ai/open-source/overview).

- **Quickstart**: Integrate Mem0 in a few lines of code
- **Playground**: Mem0 in action at https://app.mem0.ai/playground
- **Examples**: See what you can build with Mem0

## Need help?

If you have any questions, please feel free to reach out to us using one of the following methods:

- **Discord**: Join our community at https://mem0.dev/DiD
- **GitHub**: Ask questions on GitHub at https://github.com/mem0ai/mem0/discussions/new?category=q-a
- **Support**: Talk to founders at https://cal.com/taranjeetio/meet
