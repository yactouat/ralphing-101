# ralphing-101

Getting up to speed with code automation in the community.

---

## Ralph Wiggum Technique

This repo includes a minimal, beginner-friendly demo of the **Ralph Wiggum Technique**: an iterative loop where an AI agent is repeatedly given a task until it succeeds. The main idea is to avoid **context rot** by giving each iteration a **fresh context** — only the current task and the latest feedback — instead of a long history of failed attempts.

### What the script does

**`ralph_wiggum_technique.py`** uses a plain `while`-loop with [LangChain-Ollama](https://github.com/langchain-ai/langchain/tree/master/libs/partners/ollama) to implement the loop. On each iteration:

1. **Read spec** — loads `PRD.md` to understand the task requirements.
2. **Load progress** — reads `out/ralph_wiggum_technique/progress.txt` (persistent scratchpad).
3. **Generate code** — a fresh `ChatOllama` instance (no shared state) writes or fixes `analyze_news()`.
4. **Run tests** — 10 examples (7 finance, 3 non-finance) are run against the generated function:
   - Finance examples → result must be a structured Pydantic object with `sentiment` and `impact_score`.
   - Non-finance examples → `analyze_news()` must **raise an exception** (per PRD: off-topic noise raises an exception, not returns a sentinel).
5. **Update scratchpad** — test results, a LLM-generated failure analysis, and a rolling summary are written back to `progress.txt` for the next iteration.
6. **Loop or end** — if all 10 tests pass the loop exits; otherwise the attempt counter increments and the loop starts again with fresh context.

The loop is capped at **10 attempts** so it never runs forever.

Key design points the script is annotated with:
- Each `ChatOllama` instance is created fresh per iteration — no shared in-memory state between attempts.
- `progress.txt` is the only memory bridge across iterations (persistent scratchpad, not chat history).
- A modern LangChain API reference (`DOCS` constant) is injected into every prompt to prevent the model from using deprecated APIs.

### Prerequisites

- **Python 3.12+**
- **Ollama** installed and running, with the `llama3.1` model pulled:
  ```bash
  ollama run llama3.1
  ```
  (Run once so the model is available; you can stop it with Ctrl+C and keep Ollama running in the background.)
  Tolkien's *The Hobbit*.
- Python dependencies: use the project **venv** and install from the requirements file:
  ```bash
  source venv/bin/activate
  pip install -r requirements.txt
  ```
  Or run scripts with the venv interpreter: `venv/bin/python <script>.py`.

### How to run

From the project root, activate the venv (or use `venv/bin/python`):

```bash
source venv/bin/activate
```

- **Ralph Wiggum loop (default)** — iterative code-writing demo:
  ```bash
  python ralph_wiggum_technique.py
  ```
  Generated code is written to `out/ralph_wiggum_technique/generated.py`; the scratchpad is at `out/ralph_wiggum_technique/progress.txt`.

- **Financial analyst demo** — run the generated analyst on example inputs:
  ```bash
  python ralph_wiggum_technique.py --analyst
  ```
  (Run the loop first so `generated.py` exists.)

### Project structure

```
PRD.md                                   # Task specification (input to the loop)
ralph_wiggum_technique.py                # The loop + test runner + analyst demo
out/
  ralph_wiggum_technique/
    generated.py                         # Latest AI-generated analyze_news() script
    progress.txt                         # Persistent scratchpad (attempts, failures, summaries)
```

### example output

```bash
python3 ralph_wiggum_technique.py 
Ralph Wiggum Technique — while-loop, fresh LLM each iteration

Task spec   : PRD.md
Output dir  : out/ralph_wiggum_technique
Progress log: out/ralph_wiggum_technique/progress.txt
Max attempts: 10
Test suite  : 10 examples (7 finance, 3 non-finance)

=== Attempt 1/10 ===
Generating code (fresh LLM instance)...
Code written → out/ralph_wiggum_technique/generated.py
Running 10 tests...
Generating failure analysis (fresh LLM instance)...
Generating progress summary (fresh LLM instance)...
Result: FAILED
Feedback:
7/10 tests failed:
  [1] Example 1 (finance): EXCEPTION — ValueError: Off-topic noise detected: 1 validation error for Generation
text
  Input should be a valid string [type=string_type, input_value=NewsAnalysis(sentiment='N...cle summary goes here.'), input_type=NewsAnalysis]
    For further information visit https://errors.pydantic.dev/2.12/v/string_type
  Article: U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, ...
  ...

=== Attempt 2/10 ===
Generating code (fresh LLM instance)...
Code written → out/ralph_wiggum_technique/generated.py
Running 10 tests...
Generating failure analysis (fresh LLM instance)...
Generating progress summary (fresh LLM instance)...
Result: FAILED
Feedback:
7/10 tests failed:
  [1] Example 1 (finance): EXCEPTION — ValueError: Off-topic noise detected: Invalid response from LLM model
  Article: U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, ...
  [2] Example 2 (finance): EXCEPTION — ValueError: Off-topic noise detected: Invalid response from LLM model
  Article: Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery's studios ...
  [3] Example 3 (finance): EXCEPTION — ValueError: Off-topic noise detected: Invalid response from LLM model
  ...

=== Attempt 3/10 ===
Generating code (fresh LLM instance)...
Code written → out/ralph_wiggum_technique/generated.py
Running 10 tests...
Generating failure analysis (fresh LLM instance)...
Generating progress summary (fresh LLM instance)...
Result: FAILED
Feedback:
7/10 tests failed:
  [1] Example 1 (finance): EXCEPTION — ValueError: Invalid response from LLM model
  Article: U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, ...
  [2] Example 2 (finance): EXCEPTION — ValueError: Invalid response from LLM model
  Article: Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery's studios ...
  [3] Example 3 (finance): EXCEPTION — ValueError: Invalid response from LLM model
  ...

=== Attempt 4/10 ===
Generating code (fresh LLM instance)...
Code written → out/ralph_wiggum_technique/generated.py
Running 10 tests...
Generating failure analysis (fresh LLM instance)...
Generating progress summary (fresh LLM instance)...
Result: FAILED
Feedback:
7/10 tests failed:
  [1] Example 1 (finance): EXCEPTION — ValueError: Invalid response from LLM model
  Article: U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, ...
  [2] Example 2 (finance): EXCEPTION — ValueError: Invalid response from LLM model
  Article: Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery's studios ...
  [3] Example 3 (finance): EXCEPTION — ValueError: Invalid response from LLM model
  ...

=== Attempt 5/10 ===
Generating code (fresh LLM instance)...
Code written → out/ralph_wiggum_technique/generated.py
Running 10 tests...
Generating failure analysis (fresh LLM instance)...
Generating progress summary (fresh LLM instance)...
Result: FAILED
Feedback:
7/10 tests failed:
  [1] Example 1 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, ...
  [2] Example 2 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery's studios ...
  [3] Example 3 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  ...

=== Attempt 6/10 ===
Generating code (fresh LLM instance)...
Code written → out/ralph_wiggum_technique/generated.py
Running 10 tests...
Generating failure analysis (fresh LLM instance)...
Generating progress summary (fresh LLM instance)...
Result: FAILED
Feedback:
7/10 tests failed:
  [1] Example 1 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, ...
  [2] Example 2 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery's studios ...
  [3] Example 3 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  ...

=== Attempt 7/10 ===
Generating code (fresh LLM instance)...
Code written → out/ralph_wiggum_technique/generated.py
Running 10 tests...
Generating failure analysis (fresh LLM instance)...
Generating progress summary (fresh LLM instance)...
Result: FAILED
Feedback:
7/10 tests failed:
  [1] Example 1 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, ...
  [2] Example 2 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery's studios ...
  [3] Example 3 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  ...

=== Attempt 8/10 ===
Generating code (fresh LLM instance)...
Code written → out/ralph_wiggum_technique/generated.py
Running 10 tests...
Generating failure analysis (fresh LLM instance)...
Generating progress summary (fresh LLM instance)...
Result: FAILED
Feedback:
7/10 tests failed:
  [1] Example 1 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, ...
  [2] Example 2 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery's studios ...
  [3] Example 3 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  ...

=== Attempt 9/10 ===
Generating code (fresh LLM instance)...
Code written → out/ralph_wiggum_technique/generated.py
Running 10 tests...
Generating failure analysis (fresh LLM instance)...
Generating progress summary (fresh LLM instance)...
Result: FAILED
Feedback:
7/10 tests failed:
  [1] Example 1 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, ...
  [2] Example 2 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery's studios ...
  [3] Example 3 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  ...

=== Attempt 10/10 ===
Generating code (fresh LLM instance)...
Code written → out/ralph_wiggum_technique/generated.py
Running 10 tests...
Generating failure analysis (fresh LLM instance)...
Generating progress summary (fresh LLM instance)...
Result: FAILED
Feedback:
7/10 tests failed:
  [1] Example 1 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, ...
  [2] Example 2 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  Article: Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery's studios ...
  [3] Example 3 (finance): EXCEPTION — ValueError: Error analyzing news article: Invalid response from LLM model
  ...


Max attempts (10) reached. Last feedback in out/ralph_wiggum_technique/progress.txt.

Progress log : out/ralph_wiggum_technique/progress.txt
Generated code: out/ralph_wiggum_technique/generated.py
```

---

## ralph.sh — Claude Code variant

**`ralph.sh`** is a shell-based version of the same iterative loop, but uses **Claude Code** (the `claude` CLI) as the agent instead of Ollama/LangChain. The structure mirrors the Python approach: read the spec, generate code, validate with pytest, repeat until done.

### How it works

On each iteration the loop:

1. **Checks for completion** — if `out/ralph/progress.txt` contains the word `DONE`, the loop exits.
2. **Calls Claude Code** — passes `PRD.md` and `progress.txt` as context, asks it to write `out/ralph/generated.py` and update the scratchpad.
3. **Runs the external test suite** — `pytest tests/test_analyze_news.py -v` acts as the hard-truth validation gate.
4. **Loops or ends** — if tests pass the step is validated; on failure, the next iteration feeds the updated scratchpad back to Claude Code.

Key design differences from the Python version:

| | `ralph_wiggum_technique.py` | `ralph.sh` |
|---|---|---|
| Agent | Ollama (`llama3.1`) via LangChain | Claude Code CLI (`claude`) |
| Loop | Python `while` (max 10 attempts) | Bash `while true` (until `DONE`) |
| Validation | Inline Python test runner | External `pytest` subprocess |
| Output dir | `out/ralph_wiggum_technique/` | `out/ralph/` |

### Prerequisites

- **Claude Code** installed and authenticated:
  ```bash
  claude --version   # should print a version number
  ```
- Python venv with `pytest` available (same venv as above):
  ```bash
  source venv/bin/activate
  pip install -r requirements.txt
  ```

### How to run

From the project root:

```bash
chmod +x ralph.sh   # only needed once
./ralph.sh
```

Generated code is written to `out/ralph/generated.py`; the scratchpad is at `out/ralph/progress.txt`.

### Project structure (ralph.sh additions)

```
ralph.sh                        # The shell loop (Claude Code variant)
tests/
  test_analyze_news.py          # External pytest suite (reads out/ralph/generated.py)
out/
  ralph/
    generated.py                # Latest AI-generated analyze_news() script
    progress.txt                # Persistent scratchpad (updated each iteration)
```

---

### Test suite

The 10 test examples cover:

| # | Content | Expected behaviour |
|---|---------|-------------------|
| 1 | S&P 500 / Nasdaq daily gains | structured object |
| 2 | Netflix acquisition of Warner Bros. | structured object |
| 3 | Crude oil at $71/barrel | structured object |
| 4 | CoStar Q4 earnings + buyback | structured object |
| 5 | Apple record Q4 2025 earnings | structured object |
| 6 | Fed signals 25bp rate cut | structured object |
| 7 | Amazon $2.5B healthcare AI deal | structured object |
| 8 | Beef Bourguignon recipe | raise exception |
| 9 | Warriors vs Celtics NBA result | raise exception |
| 10 | Nolan film at Cannes | raise exception |

### example output

```bash
./ralph.sh 
Starting the Ralph Wiggum loop. I'm helping!
========================================
Starting Iteration 1
========================================
All 10 tests pass and the task is complete. Here's a summary of what was done:

**Step executed:** Created `out/ralph/generated.py` implementing the `analyze_news()` function.

**Implementation:**
- Two-step LangChain pipeline using Ollama `llama3.1`
- **Step 1 — Classification:** `ChatOllama + StrOutputParser` chain to determine if the article is finance-related. Non-finance articles raise `ValueError`
- **Step 2 — Extraction:** `ChatOllama.with_structured_output(NewsAnalysis)` chain returns a validated Pydantic object with `sentiment`, `impact_score`, `ticker_symbols`, and `summary`

**Bug fixed during iteration:** The initial classifier rejected the Amazon M&A article (finance-7) because the model focused on "healthcare startup" rather than the acquisition itself. Fixed by expanding the system prompt to explicitly list M&A, corporate acquisitions, and company spending as always finance-related topics.

**Final result:** 10/10 tests passed — `STATUS: COMPLETE`
Iteration 1 finished. Sleeping for 5 seconds...
========================================
Starting Iteration 2
========================================
Task completed successfully! Exiting Ralph loop.
```