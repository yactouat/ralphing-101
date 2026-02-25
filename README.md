# ralphing-101

Getting up to speed with code automation in the community.

---

## 1. Ralph Wiggum Technique

This repo includes a minimal, beginner-friendly demo of the **Ralph Wiggum Technique**: an iterative loop where an AI agent is repeatedly given a task until it succeeds. The main idea is to avoid **context rot** by giving each iteration a **fresh context** — only the current task and the latest feedback — instead of a long history of failed attempts.

### What the script does

- **`1_ralph_wiggum_technique.py`** uses [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain-Ollama](https://github.com/langchain-ai/langchain/tree/master/libs/partners/ollama) to implement the loop:
  1. **Write Code** — An LLM (Ollama `qwen3`) gets only the task description and any feedback, and outputs Python code (no chat history is passed).
  2. **Run Tests** — A simulated checker verifies the code (e.g. that it contains `def` and `return`). If it fails, feedback is produced.
  3. **Loop or End** — If tests pass, the graph ends; otherwise the attempt count is incremented, feedback is stored, and the graph goes back to **Write Code** with the same flat state (fresh context again).
- The loop is capped at **10 attempts** so the demo never runs forever.

The script is heavily commented to highlight where the fresh context is created and how we avoid context rot (e.g. no growing message list in state).

### Prerequisites

- **Python 3.12+**
- **Ollama** installed and running, with the `qwen3` model pulled:
  ```bash
  ollama run qwen3
  ```
  (Run once so the model is available; you can stop it with Ctrl+C and keep Ollama running in the background.)
- Python dependencies:
  Install from the project’s requirements file:
  ```bash
  pip install -r requirements.txt
  ```

### How to run

From the project root:

- **Ralph Wiggum loop (default)** — iterative code-writing demo:
  ```bash
  python 1_ralph_wiggum_technique.py
  ```
  You should see the task, then the final state (attempts used, last feedback, and the generated code) after the graph finishes. Output code is written to `out/ralph_wiggum_technique/generated.py`.

- **Financial analyst demo** — Ralph-Proof structured extraction (off-topic / parse handling):
  ```bash
  python 1_ralph_wiggum_technique.py --analyst
  ```
  This runs the analyst chain instead of the loop; no code file is written.

  #### example script produced with the 1st ralphing script

  ```python
  """
Financial News Analysis Module

This module provides a function to analyze news articles for financial insights using
LangChain's structured output pattern with Ollama's ChatOllama model. The analysis
returns a structured Pydantic object if the article is finance-related, otherwise
returns 'NOT_FINANCE'.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class FinancialAnalysis(BaseModel):
    """
    Pydantic model for structured financial analysis output.
    
    Attributes:
        sentiment (str): Market sentiment - Bullish/Bearish/Neutral
        impact_score (int): 1-10 scale for market impact
        ticker_symbols (List[str]): Relevant stock tickers
        summary (str): Summary of key financial insights
    """
    sentiment: str = Field(..., enum=["Bullish", "Bearish", "Neutral"])
    impact_score: int = Field(..., ge=1, le=10)
    ticker_symbols: List[str] = Field(...)
    summary: str = Field(...)

# Initialize Ollama chat model with temperature for balanced responses
llm = ChatOllama(model="qwen3", temperature=0.2)
# Create structured output runnable for FinancialAnalysis
structured_llm = llm.with_structured_output(FinancialAnalysis)

def analyze_news(article: str) -> Any:
    """
    Analyze a news article for financial insights.
    
    Args:
        article (str): The news article text to analyze
        
    Returns:
        FinancialAnalysis: Structured analysis if finance-related
        str: 'NOT_FINANCE' if no financial angle detected
    """
    # First check if article contains finance-related keywords
    if not is_finance_related(article):
        return "NOT_FINANCE"
    
    # Create prompt template to guide the analysis
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Analyze this news article: {article}")
    ])
    
    # Chain prompt with structured output model
    analysis_chain = prompt | structured_llm
    
    # Execute analysis and return result
    return analysis_chain.invoke({"article": article})

def is_finance_related(text: str) -> bool:
    """
    Check if text contains finance-related keywords.
    
    Args:
        text (str): Text to analyze for financial context
        
    Returns:
        bool: True if finance-related keywords are found
    """
    # Comprehensive list of finance-related keywords
    finance_keywords = {
        "finance", "stock", "market", "investment", "budget", "currency", 
        "interest", "portfolio", "equity", "dividend", "bond", "recession", 
        "inflation", "economic", "growth", "sector", "industry", "asset", 
        "liquidity", "capital", "profit", "loss", "share", "price", "volume"
    }
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check for any matching keyword
    return any(keyword in text_lower for keyword in finance_keywords)

# Example usage:
# result = analyze_news("Stock market hits new highs after Q3 earnings reports")
# if isinstance(result, FinancialAnalysis):
#     print(f"Sentiment: {result.sentiment}, Impact: {result.impact_score}")
# else:
#     print("No financial analysis found")
```
