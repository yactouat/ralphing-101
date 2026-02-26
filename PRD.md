*   **The Goal:** Build an LLM-powered analyst that turns messy news into high-signal data.
*   **Input:** Raw news items (breaking articles, market gossip).
*   **Output:** A structured object containing:
    *   `sentiment` (Bullish/Bearish/Neutral)
    *   `impact_score` (1-10)
    *   `ticker_symbols` (List)
    *   `summary` (Short text)
*   **Exception handling:** Handling off-topic noise (e.g., cooking recipes) by throwing an exception.
*   **LLM Backend:** Ollama running the `llama3.1` local model. Use LangChain's Ollama integration to call the model.
*   **Dependencies:** No libraries other than LangChain and Pydantic should be used to complete this task.