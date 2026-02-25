#!/usr/bin/env python3
"""
Spec Alchemy — From "Code Vomit" to Objective Constraints + TDD.

This script demonstrates the evolution beyond the naive Ralph Wiggum loop:
instead of a weak success contract ("does the code run?"), we use STRICT
OBJECTIVE TESTS and a PERSISTENT SCRATCHPAD to force the AI to understand
business logic nuance. The bottleneck shifts from "writing code" to
"writing tests and specs."

Concepts demonstrated:
  • Spec Alchemy: Constraints act as the "mold" — the AI must satisfy
    explicit test cases (e.g. AAPL in ticker_symbols, NOT_FINANCE for
    recipes, Bullish for commodity price moves). Without these, the model
    would repeatedly generate code that "runs" but misclassifies edge cases.
  • Context Rot (the malloc/free problem): Long conversation history
    or just passing the previous code to the LLM dilutes important failure details.
    We avoid this by using an external memory bank (progress.txt):
    each run_tests appends attempt number + exact test results; write_code
    reads the ENTIRE scratchpad and injects it into the prompt. The LLM
    sees a concise log of what failed and why, not a bloated dialogue or half-baked code.
  • TDD bucket: The run_tests node runs a strict test suite. Feedback
    states which test passed/failed and the exact expected vs actual output,
    so the next write_code iteration can fix specific failures.

Requirements: Python 3.12+, langgraph, langchain-ollama, Ollama (qwen3).

How to run (use the project venv):
  source venv/bin/activate   # or: venv/bin/python
  python 2_ralfing_with_better_specs.py              # run the spec-alchemy loop
  python 2_ralfing_with_better_specs.py --analyst    # run analyst on examples
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# 1. STATE — What the graph carries between nodes
# ---------------------------------------------------------------------------
# We use a TypedDict for clarity and tooling. The key difference from the
# "Code Vomit" script: feedback is now RICH (per-test pass/fail, expected
# vs actual). The scratchpad (progress.txt) is the persistent memory;
# we do not store it in state — we read it from disk in write_code.


class SpecAlchemyState(TypedDict):
    """State for the Spec Alchemy loop. Feedback is detailed test results."""

    task_description: str
    """The original task (financial analyst with NOT_FINANCE rule)."""

    current_code: str
    """Latest generated code. Overwritten each attempt."""

    feedback: str
    """Detailed feedback from run_tests: which test passed/failed, expected vs actual."""

    attempt_number: int
    """Current attempt index. Used for cap (MAX_ATTEMPTS) and for progress.txt."""

    tests_passed: bool
    """True when all three objective tests pass."""


# ---------------------------------------------------------------------------
# 2. CONFIG — Paths, model, and loop cap
# ---------------------------------------------------------------------------
MAX_ATTEMPTS = 10
OLLAMA_MODEL = "qwen3"
OUTPUT_DIR = Path("out/ralfing_with_better_specs")
PROGRESS_FILE = OUTPUT_DIR / "progress.txt"
GENERATED_FILE = OUTPUT_DIR / "generated.py"

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

# LangChain structured output docs: same as in script 1, so the model uses
# current APIs (with_structured_output, ChatOllama, Pydantic).
LANGCHAIN_STRUCTURED_OUTPUT_DOCS = """
LANGCHAIN STRUCTURED OUTPUT — use this exact pattern (do NOT use legacy langchain.chains, langchain.llms.Ollama, or PromptTemplateChain):

Required imports:
  from langchain_ollama import ChatOllama
  from langchain_core.prompts import ChatPromptTemplate
  from pydantic import BaseModel, Field

Pattern:
  1. Define a Pydantic class (e.g. FinancialAnalysis) with: sentiment (Bullish/Bearish/Neutral), impact_score (1-10), ticker_symbols (list of str), summary (str).
  2. Create a CHAT model: llm = ChatOllama(model="qwen3", temperature=0.2)
  3. structured_llm = llm.with_structured_output(FinancialAnalysis)
  4. result = structured_llm.invoke([HumanMessage(content=article)])
  If the input is NOT finance-related, return the exact string "NOT_FINANCE".
"""


# ---------------------------------------------------------------------------
# 3. OBJECTIVE TEST CASES — The "TDD bucket" (the mold)
# ---------------------------------------------------------------------------
# Each test has: name, input text, and a checker that returns (passed, message).
# The message describes what was expected vs what was actually returned.
# This is what gets written to progress.txt and fed back to the LLM.


# New helper function to check if the result is a Pydantic object with the required attributes.
# This is technical validation to ensure the result is a valid Pydantic object.
# It is used in the _check_test_1_standard_finance and _check_test_3_commodity_bullish functions.
def _is_pydantic_result(obj: Any) -> bool:
    """True if obj looks like our Pydantic analysis (has sentiment, impact_score, ticker_symbols, summary)."""
    if obj is None:
        return False
    return (
        hasattr(obj, "sentiment")
        and hasattr(obj, "impact_score")
        and hasattr(obj, "ticker_symbols")
        and hasattr(obj, "summary")
    )


def _check_test_1_standard_finance(actual: Any) -> tuple[bool, str]:
    """Test 1: Apple Q4 earnings -> Pydantic object with ticker_symbols containing AAPL."""
    if actual is None:
        return False, "Expected a Pydantic analysis object; got None."
    if not _is_pydantic_result(actual):
        return (
            False,
            f"Expected a Pydantic analysis object (sentiment, impact_score, ticker_symbols, summary). "
            f"Got type {type(actual).__name__}: {repr(actual)[:200]}.",
        )
    tickers = getattr(actual, "ticker_symbols", None)
    if not isinstance(tickers, list):
        return (
            False,
            f"Expected ticker_symbols to be a list. Got {type(tickers).__name__}.",
        )
    aapl = "AAPL" in tickers or any(
        (isinstance(t, str) and t.upper() == "AAPL" for t in tickers)
    )
    if not aapl:
        return (
            False,
            f"Expected ticker_symbols to include 'AAPL'. Got: {tickers}. "
            f"Actual: {actual}",
        )
    return True, "Pass: Pydantic object with AAPL in ticker_symbols."


def _check_test_2_not_finance(actual: Any) -> tuple[bool, str]:
    """Test 2: Beef Bourguignon recipe -> exact string NOT_FINANCE."""
    if actual == "NOT_FINANCE":
        return True, "Pass: returned exact string 'NOT_FINANCE'."
    return (
        False,
        f"Expected exact string 'NOT_FINANCE' for off-topic input. "
        f"Got type {type(actual).__name__}, value: {repr(actual)[:200]}.",
    )


def _check_test_3_commodity_bullish(actual: Any) -> tuple[bool, str]:
    """Test 3: Crude oil higher -> Pydantic object, sentiment Bullish."""
    if actual is None:
        return False, "Expected a Pydantic analysis object; got None."
    if not _is_pydantic_result(actual):
        return (
            False,
            f"Expected a Pydantic analysis object for commodity/finance news. "
            f"Got type {type(actual).__name__}: {repr(actual)[:200]}.",
        )
    sentiment = getattr(actual, "sentiment", None)
    if sentiment is None:
        return False, "Expected 'sentiment' attribute. Got None."
    if str(sentiment).strip().lower() != "bullish":
        return (
            False,
            f"Expected sentiment 'Bullish' for 'crude oil traded higher'. "
            f"Got: {repr(sentiment)}. Full object: {actual}",
        )
    return True, "Pass: Pydantic object with sentiment Bullish."


# Test case definitions: (short_name, input_text, checker)
TEST_CASES = [
    (
        "Standard Finance (AAPL)",
        "Apple announces record Q4 earnings.",
        _check_test_1_standard_finance,
    ),
    (
        "Irrelevant/Noise (NOT_FINANCE)",
        "To make Beef Bourguignon, simmer a chuck roast in Pinot Noir.",
        _check_test_2_not_finance,
    ),
    (
        "Ambiguous/Commodity (Bullish)",
        "Crude oil traded higher this morning, breaking the $71 per barrel mark...",
        _check_test_3_commodity_bullish,
    ),
]


# ---------------------------------------------------------------------------
# 4. RUN GENERATED CODE — Load module and call analyze_news
# ---------------------------------------------------------------------------
def _run_analyze_news(generated_path: Path, text: str) -> tuple[Any, str | None]:
    """
    Load generated.py and call analyze_news(text). Return (result, None) on success,
    (None, error_message) on load/call failure.
    """
    if not generated_path.exists():
        return None, "Generated file does not exist."
    try:
        spec = importlib.util.spec_from_file_location(
            "generated_spec_alchemy", generated_path
        )
        if spec is None or spec.loader is None:
            return None, "Could not load generated module."
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_spec_alchemy"] = module
        spec.loader.exec_module(module)

        analyze_news = getattr(module, "analyze_news", None)
        if analyze_news is None or not callable(analyze_news):
            return None, "Generated code has no callable 'analyze_news'."
        result = analyze_news(text)
        return result, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 5. NODE: RUN TESTS — Strict test suite + append to progress.txt
# ---------------------------------------------------------------------------
# This is the "TDD bucket": we run three objective tests and produce
# explicit feedback (which passed, which failed, expected vs actual).
# That feedback is both returned in state AND appended to progress.txt
# so that the next write_code can read it (Context Rot fix).


def run_tests(state: SpecAlchemyState) -> dict[str, str | int | bool]:
    """
    Write current code to out/ralfing_with_better_specs/generated.py, run the
    three objective tests, build detailed feedback, append to progress.txt,
    and return updated state (feedback, attempt_number, tests_passed).
    """
    code = state["current_code"]
    attempt = state["attempt_number"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_FILE.write_text(code, encoding="utf-8")

    results: list[str] = []
    all_passed = True

    for name, input_text, checker in TEST_CASES:
        actual, load_err = _run_analyze_news(GENERATED_FILE, input_text)
        if load_err:
            results.append(f"  [{name}] ERROR: {load_err}")
            all_passed = False
            continue
        passed, msg = checker(actual)
        if passed:
            results.append(f"  [{name}] PASS: {msg}")
        else:
            results.append(f"  [{name}] FAIL: {msg}")
            all_passed = False

    feedback_body = "Test results:\n" + "\n".join(results)

    # Append to progress.txt (persistent scratchpad — solves Context Rot)
    progress_line = f"\n--- Attempt {attempt} ---\n{feedback_body}\n"
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(progress_line)

    if all_passed:
        return {"tests_passed": True, "feedback": feedback_body}
    return {
        "feedback": feedback_body,
        "attempt_number": attempt + 1,
        "tests_passed": False,
    }


# ---------------------------------------------------------------------------
# 6. NODE: WRITE CODE — Inject progress.txt so the LLM avoids repeating mistakes
# ---------------------------------------------------------------------------
# Spec Alchemy upgrade: we do NOT rely on conversational  or just passing the previous code to the LLM.
# We read the ENTIRE progress.txt and inject it into the prompt. The LLM is told
# to use this scratchpad to plan the next fix and avoid repeating past failures. 
# This replaces context rot from long chat history or just passing the previous code to the LLM with a
# concise, persistent log of previous attempts and test results.


def write_code(state: SpecAlchemyState) -> dict[str, str]:
    """
    Ask the LLM to produce or fix Python code. The prompt includes:
    - Task description
    - Full content of progress.txt (scratchpad of previous attempts and test results)
    - Optional: current code from state or disk for retries
    - Last feedback from run_tests
    """
    task = state["task_description"]
    feedback = state["feedback"]
    attempt = state["attempt_number"]

    print(f"Attempt {attempt} — generating code...")

    # Load existing code from state or disk so we can ask the LLM to fix it
    existing_code = (state.get("current_code") or "").strip()
    if not existing_code and GENERATED_FILE.exists():
        existing_code = GENERATED_FILE.read_text(encoding="utf-8").strip()

    # SPEC ALCHEMY / CONTEXT ROT FIX: Read the entire scratchpad from disk.
    # This is the "external memory bank" — we do not pass raw chat history.
    scratchpad = ""
    if PROGRESS_FILE.exists():
        scratchpad = PROGRESS_FILE.read_text(encoding="utf-8").strip()

    # Build user message: task + scratchpad + (optional) current code + feedback
    parts = [f"Task:\n{task}\n"]

    if scratchpad:
        parts.append(
            "Here is your scratchpad (progress.txt) showing your previous attempts "
            "and test failures. Use this to plan your next fix and avoid repeating mistakes.\n\n"
            f"--- progress.txt ---\n{scratchpad}\n--- end progress.txt ---\n"
        )

    if existing_code:
        parts.append(
            "Current script (fix or enhance it; output the full improved code):\n\n"
            f"```python\n{existing_code}\n```\n\n"
        )

    parts.append(f"This is attempt {attempt}.")
    if feedback:
        parts.append(f"Latest test results:\n{feedback}\n")
    parts.append(
        "Output only valid Python code. No markdown code fences, no explanation."
    )

    user_content = "\n".join(parts)

    system_content = (
        "You are a Python programmer. The generated script must define a function "
        "analyze_news(article: str) that returns either a Pydantic object (sentiment, "
        "impact_score, ticker_symbols, summary) or the exact string 'NOT_FINANCE' for "
        "non-finance input. Use the scratchpad to avoid repeating past failures. "
        "Reply with only executable Python code, no markdown.\n\n"
        + LANGCHAIN_STRUCTURED_OUTPUT_DOCS
    )
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    code = raw.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return {"current_code": code}


# ---------------------------------------------------------------------------
# 7. ROUTING — After run_tests: retry or end
# ---------------------------------------------------------------------------
def route_after_tests(state: SpecAlchemyState) -> str:
    if state.get("tests_passed", False):
        return "end"
    if state["attempt_number"] >= MAX_ATTEMPTS:
        return "end"
    return "retry"


# ---------------------------------------------------------------------------
# 8. BUILD THE GRAPH
# ---------------------------------------------------------------------------
builder = StateGraph(SpecAlchemyState)
builder.add_node("write_code", write_code)
builder.add_node("run_tests", run_tests)
builder.add_edge(START, "write_code")
builder.add_edge("write_code", "run_tests")
builder.add_conditional_edges(
    "run_tests",
    route_after_tests,
    {"retry": "write_code", "end": END},
)
graph = builder.compile()


# ---------------------------------------------------------------------------
# 9. ANALYST DEMO — Run generated analyze_news on example inputs
# ---------------------------------------------------------------------------
def run_analyst_chain(news_input: str) -> Any:
    """Load out/ralfing_with_better_specs/generated.py and call analyze_news(news_input)."""
    if not GENERATED_FILE.exists():
        return (
            "Spec Alchemy: No generated code. Run the loop first: "
            "python 2_ralfing_with_better_specs.py"
        )
    try:
        spec = importlib.util.spec_from_file_location(
            "generated_spec_alchemy", GENERATED_FILE
        )
        if spec is None or spec.loader is None:
            return "Error: Could not load generated module."
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_spec_alchemy"] = module
        spec.loader.exec_module(module)
        analyze_news = getattr(module, "analyze_news", None)
        if analyze_news is None or not callable(analyze_news):
            return "Error: No callable 'analyze_news' in generated code."
        return analyze_news(news_input)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def run_analyst_demo() -> None:
    """Run the financial analyst on the same examples as the test cases + extras."""
    print("Spec Alchemy — Financial Analyst (generated code)\n")
    examples = [
        "U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, as Anthropic launched new artificial intelligence integrations aimed at enterprise customers, driving software subscription trends upward."
        "Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery’s studios and HBO Max is expected to close ahead of schedule in Q3 2026, shifting the landscape of streaming consolidation and media market caps.",
        "Crude oil traded higher this morning, breaking the $71 per barrel mark, amid rising tensions between the U.S. and Iran ahead of the highly anticipated nuclear talks in Geneva.",
        "CoStar Group (CSGP) posted full-year 2025 results showing 19% revenue growth and announced a massive $700 million share repurchase program for 2026, signaling strong capital return for investors.",
        "To make a traditional Beef Bourguignon, slowly braise a chuck roast in a full-bodied Pinot Noir with pearl onions, mushrooms, and thick-cut bacon. Let it simmer at 160°C for at least three hours until the meat is fork-tender.",
    ]
    for i, news in enumerate(examples, 1):
        print(f"--- Example {i} ---")
        print("News:", news[:70] + "..." if len(news) > 70 else news)
        result = run_analyst_chain(news)
        print("Result:", result)
        print()


# ---------------------------------------------------------------------------
# 10. LOOP DEMO — Run the Spec Alchemy graph
# ---------------------------------------------------------------------------
def run_loop_demo() -> None:
    """Run the Spec Alchemy loop: write_code → run_tests (with progress.txt) until pass or MAX_ATTEMPTS."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # progress.txt is never cleared here — it persists across runs (true scratchpad / Context Rot fix).

    existing_code = ""
    initial_feedback = ""
    if GENERATED_FILE.exists():
        existing_code = GENERATED_FILE.read_text(encoding="utf-8").strip()
        initial_feedback = (
            "A previous version exists. Fix it to pass all three objective tests."
        )

    task = (
        "Write a function named analyze_news(article: str) that returns either:\n"
        "  (A) A Pydantic object with: sentiment (Bullish/Bearish/Neutral), impact_score (1-10), "
        "ticker_symbols (list of str), summary (str); OR\n"
        "  (B) The exact string 'NOT_FINANCE' when the input is not finance-related (e.g. cooking, sports).\n"
        "Rules: Use ChatOllama and with_structured_output. For commodity/market news (e.g. crude oil price), "
        "return the Pydantic object (e.g. Bullish), not NOT_FINANCE. For Apple earnings, include 'AAPL' in ticker_symbols. "
        "Heavily comment the script: docstrings and section comments."
    )

    initial_state: SpecAlchemyState = {
        "task_description": task,
        "current_code": existing_code,
        "feedback": initial_feedback,
        "attempt_number": 1,
        "tests_passed": False,
    }

    print("Spec Alchemy — Objective tests + progress.txt scratchpad\n")
    print("Task:", task[:200] + "...")
    print("Max attempts:", MAX_ATTEMPTS)
    print("Output:", GENERATED_FILE)
    print("Scratchpad:", PROGRESS_FILE, "\n")

    final_state = graph.invoke(initial_state)

    print("--- Final state ---")
    print("Attempts used:", final_state["attempt_number"])
    print("Tests passed:", final_state["tests_passed"])
    print("Last feedback:\n", final_state["feedback"] or "(none)")
    print("\nGenerated code written to:", GENERATED_FILE)


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spec Alchemy: TDD + progress scratchpad"
    )
    parser.add_argument(
        "--analyst",
        action="store_true",
        help="Run the financial analyst on example inputs instead of the loop",
    )
    args = parser.parse_args()
    if args.analyst:
        run_analyst_demo()
    else:
        run_loop_demo()


if __name__ == "__main__":
    main()
