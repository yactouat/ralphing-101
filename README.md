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
