# Architectural & Technology Decisions

## DEC-001: Pydantic Settings for Configuration Management
- **Context:** Need a safe way to load API keys and paths across multiple modules.
- **Decision:** Use `pydantic-settings` to enforce schema validation at startup.
- **Why:** Halts application startup immediately if required keys (like `GROQ_API_KEY`) are missing, preventing late runtime failures.

## DEC-002: Deterministic Inference (Temperature = 0.0)
- **Context:** Financial SEC extraction requires exact numerical precision.
- **Decision:** Lock LLM temperature to `0.0`.
- **Why:** Eliminates stylistic randomness and reduces hallucination risk when parsing balance sheets and income statements.

## DEC-003: Groq + Local HuggingFace Embeddings ($0 Stack)
- **Context:** Keep production deployment and local development completely free.
- **Decision:** Use Groq Cloud free tier (`llama-3.3-70b-versatile`) for reasoning and `sentence-transformers/all-MiniLM-L6-v2` locally for embeddings.
- **Why:** High-speed inference without credit card requirements or vendor lock-in.

## DEC-004: TypedDict Schema for Graph State
- **Context:** State must be passed deterministically across multiple agentic nodes.
- **Decision:** Use a strictly typed `AgentState(TypedDict)` containing query decompose targets, document collections, retry counters, and hallucination flags.
- **Why:** Prevents state mutation bugs, provides IDE autocompletion, and bounds retry loops to prevent infinite execution cycles.