# Model routing status

CampusReg currently has a single generation-provider implementation: the Google GenAI SDK adapter in `backend/campusreg/rag/generation.py`. `RAGSettings.gemma_model` is configurable, but the application does not maintain an OpenRouter model registry, context-capacity table, model router, or provider fallback policy.

Consequently there is no selection among Gemma 2/3/4 models, no route-reason details panel, no validated per-model token budgets, and no comparative quality/latency/cost result. This feature is not enabled in the UI. Adding model IDs alone would not establish their availability or context limits; those values must be validated against the provider and deployment account before a route is enabled.

The current single-provider failure path returns a structured generation error. It does not silently substitute another model. Do not interpret the configured model name as a measured routing decision.
