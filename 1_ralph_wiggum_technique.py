#!/usr/bin/env python3
"""
Ralph Wiggum Technique — Autonomous AI development loop with fresh context.

This script demonstrates:
1. An iterative loop where an AI agent is repeatedly given a task until it
   succeeds (each iteration gets fresh context only).
2. A "Ralph-Proof" financial analyst chain: structured extraction with
   graceful handling of off-topic input (NOT_FINANCE) and parsing errors.

Requirements: Python 3.12+, langgraph, langchain-ollama, Ollama (qwen3).

How to run (for the demo):
  python 1_ralph_wiggum_technique.py              # runs the code-generation loop
  python 1_ralph_wiggum_technique.py --analyst   # runs the analyst on example inputs
"""

# Standard library: CLI args, loading Python files dynamically, paths
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import TypedDict

# LangChain/LangGraph: local LLM (Ollama), message types, and the graph (state machine)
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# 1. STATE — What the graph remembers (and what it does NOT)
# ---------------------------------------------------------------------------
# LangGraph runs a "graph" of steps (nodes). Each node receives "state" — a
# dictionary that gets updated as we go. We use a TypedDict to declare exactly
# which keys exist and what type each is (helps readability and tooling).
#
# Important: we do NOT keep a long chat history or a list of past code attempts.
# We only keep the current snapshot. So each time we call the LLM we build the
# prompt from: task + (optional) previous script + feedback. That's "fresh
# context" — no context rot from huge conversations.


class RalphState(TypedDict):
    """State for the Ralph Wiggum loop. No message history — fresh context only."""

    task_description: str
    """The original task (e.g. 'Analyze this news for financial sentiment and impact')."""

    current_code: str
    """Latest generated code. Overwritten each attempt; not accumulated."""

    feedback: str
    """Feedback from the last test run (e.g. what failed). Empty on first attempt."""

    attempt_number: int
    """How many times we've run the write_code → run_tests cycle. Used for cap and prompts."""

    tests_passed: bool
    """True when generated.py exists and runs without error on a valid example."""


# ---------------------------------------------------------------------------
# 2. CONFIG — Failsafe and model
# ---------------------------------------------------------------------------
MAX_ATTEMPTS = 10  # Cap the loop so a beginner tutorial never runs forever.
OLLAMA_MODEL = "qwen3"
OUTPUT_DIR = Path("out/ralph_wiggum_technique")  # Where we write generated.py

# This string is passed to analyze_news() to check that generated.py runs without errors.
VALID_EXAMPLE = (
    "U.S. stocks closed higher on Tuesday, boosted by positive labor market reports."
)

# Our "brain": local LLM via Ollama. temperature=0.2 keeps outputs relatively stable.
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

# Documentation for the code-generation LLM: correct LangChain structured output syntax.
# Injected into every iteration so the model (e.g. qwen3) uses current APIs, not legacy ones.
LANGCHAIN_STRUCTURED_OUTPUT_DOCS = """
LANGCHAIN STRUCTURED OUTPUT — use this exact pattern (do NOT use legacy langchain.chains, langchain.llms.Ollama, or PromptTemplateChain):

Required imports:
  from langchain_ollama import ChatOllama
  from langchain_core.prompts import ChatPromptTemplate
  from pydantic import BaseModel, Field

Pattern:
  1. Define a Pydantic class (e.g. FinancialAnalysis) with the required fields.
  2. Create a CHAT model (ChatOllama), not a legacy LLM:
     llm = ChatOllama(model="qwen3", temperature=0.2)
  3. Get a structured runnable and invoke it with messages; the result is the Pydantic instance:
     structured_llm = llm.with_structured_output(FinancialAnalysis)
     result = structured_llm.invoke([HumanMessage(content=prompt_or_article)])
     # result is already a FinancialAnalysis instance — no json.loads or parse_obj needed.
  4. Optional: use a prompt template then invoke:
     prompt = ChatPromptTemplate.from_messages([("human", "Analyze this news article: {article}")])
     chain = prompt | structured_llm
     result = chain.invoke({"article": article_text})
"""


# ---------------------------------------------------------------------------
# 2b. FINANCIAL ANALYST — "Ralph-Proof" structured extraction
# ---------------------------------------------------------------------------
# If the input is NOT finance-related, we ask the model to output NOT_FINANCE
# and handle it gracefully (Ralph moment). Same for parsing errors.
# Uses local Ollama (qwen3) like the rest of the script.


def run_analyst_chain(news_input: str):
    """
    Run the generated code (out/ralph_wiggum_technique/generated.py): load the
    module, call analyze_news(news_input), and return the result. On missing
    file, missing analyze_news, or execution errors, return a Ralph-style
    graceful message (no ugly tracebacks for the audience).
    """
    generated_path = OUTPUT_DIR / "generated.py"
    if not generated_path.exists():
        return (
            "👔 Ralph says: 'I'm a financial analyst!' "
            "(Error: No generated code found. Run the loop first to generate it.)"
        )

    try:
        # Load generated.py as a Python module (without importing by file path in the usual way)
        spec = importlib.util.spec_from_file_location("generated", generated_path)
        if spec is None or spec.loader is None:
            return "🚨 [Ralph Technique Triggered]: Could not load generated module."
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated"] = module
        spec.loader.exec_module(module)

        # Get the analyze_news function; if missing, fail gracefully
        analyze_news = getattr(module, "analyze_news", None)
        if analyze_news is None or not callable(analyze_news):
            return (
                "👔 Ralph says: 'I'm a financial analyst!' "
                "(Error: Generated code has no callable 'analyze_news'. Graceful exit initiated.)"
            )

        return analyze_news(news_input)

    except Exception as e:
        return f"🚨 [Ralph Technique Triggered]: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 3. NODE: WRITE CODE
# ---------------------------------------------------------------------------
# This is where "fresh context" happens: we do NOT pass previous attempts
# or a long conversation. We pass only:
#   - The task description
#   - The latest feedback (if any)
#   - The current attempt number
# So the LLM sees a clean, minimal prompt every time — no context rot.
#
# RETRY LOGIC: When a previous attempt failed (or when we re-run the script and
# generated.py already exists), we pass that previous script to the LLM so it can
# *fix* or *enhance* it instead of generating from scratch. That way retries are
# informed by the last version (from state or from disk).


def write_code(state: RalphState) -> dict[str, str | int]:
    """
    Ask the LLM to produce Python code from the task and feedback only.
    Returns a partial state update (current_code, and we keep feedback/attempt as-is
    until run_tests updates them).
    """
    task = state["task_description"]
    feedback = state["feedback"]
    attempt = state["attempt_number"]

    print(f"Attempt {attempt} — generating code...")

    # RETRY: Use previous script when available — from state (same run) or from disk (e.g. re-run).
    existing_code = (state.get("current_code") or "").strip()
    if not existing_code and (OUTPUT_DIR / "generated.py").exists():
        existing_code = (
            (OUTPUT_DIR / "generated.py").read_text(encoding="utf-8").strip()
        )

    # Build one prompt. If we have existing code, ask the LLM to improve it; else ask for new code.
    if existing_code:
        base = (
            f"Task:\n{task}\n\n"
            "Here is the current script (enhance or fix it; output the full improved code):\n\n"
            "```python\n"
            f"{existing_code}\n"
            "```\n\n"
        )
        if feedback:
            user_content = (
                base
                + f"This is attempt {attempt}. Previous run failed with:\n{feedback}\n\n"
                + "Output only valid Python code. No markdown code fences, no explanation."
            )
        else:
            user_content = (
                base
                + f"This is attempt {attempt}. Output the full enhanced Python code. No markdown code fences, no explanation."
            )
    elif feedback:
        user_content = (
            f"Task:\n{task}\n\n"
            f"This is attempt {attempt}. Previous attempt failed with this feedback:\n{feedback}\n\n"
            "Output only valid Python code. No markdown code fences, no explanation."
        )
    else:
        user_content = (
            f"Task:\n{task}\n\n"
            f"This is attempt {attempt}. Output only valid Python code. No markdown code fences, no explanation."
        )

    # System message sets the "role" and includes LangChain docs on every iteration.
    system_content = (
        "You are a Python programmer. The generated script must be heavily commented: "
        "use docstrings for modules and functions, section comments for logical blocks, "
        "and inline comments where helpful. Reply with only executable Python code, no markdown.\n\n"
        + LANGCHAIN_STRUCTURED_OUTPUT_DOCS
    )
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    # Many LLMs wrap code in ```python ... ```. Strip that so we get plain executable code.
    code = raw.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    # Return only the keys we want to update in state. LangGraph merges this into the full state.
    return {"current_code": code}


# ---------------------------------------------------------------------------
# 4. NODE: RUN TESTS
# ---------------------------------------------------------------------------
# Termination conditions: (1) out/ralph_wiggum_technique/generated.py exists,
# (2) it runs without throwing when called with a valid example.


def _run_generated_with_example(generated_path: Path, example: str) -> tuple[bool, str]:
    """
    Load generated.py and call analyze_news(example). Return (True, '') on success,
    (False, error_message) on missing file, missing analyze_news, or exception.
    """
    if not generated_path.exists():
        return False, "Generated file does not exist."
    try:
        spec = importlib.util.spec_from_file_location("generated", generated_path)
        if spec is None or spec.loader is None:
            return False, "Could not load generated module."
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated"] = module
        spec.loader.exec_module(module)

        analyze_news = getattr(module, "analyze_news", None)
        if analyze_news is None or not callable(analyze_news):
            return False, "Generated code has no callable 'analyze_news'."
        analyze_news(example)  # Actually run it; any exception is caught below
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def run_tests(state: RalphState) -> dict[str, str | int | bool]:
    """
    Write current code to out/ralph_wiggum_technique/generated.py, then verify
    the file exists and runs without error when calling analyze_news(VALID_EXAMPLE).
    If it passes, set tests_passed=True and go to END; else set feedback and retry.
    """
    code = state["current_code"]
    attempt = state["attempt_number"]
    generated_path = OUTPUT_DIR / "generated.py"

    # Persist the latest code so we can run it (and so next retry can load it from disk if needed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(code, encoding="utf-8")

    passed, error_msg = _run_generated_with_example(generated_path, VALID_EXAMPLE)

    if passed:
        return {"tests_passed": True}
    # Tests failed: store the error as feedback and bump attempt so write_code can retry with context
    return {
        "feedback": error_msg,
        "attempt_number": attempt + 1,
        "tests_passed": False,
    }


# ---------------------------------------------------------------------------
# 5. ROUTING: After run_tests, continue or end?
# ---------------------------------------------------------------------------
# After run_tests we either finish (end) or send the graph back to write_code
# (retry). The state already has updated feedback and attempt_number, so the
# next write_code call gets the error message and can load the last script from
# state/disk to fix it.


def route_after_tests(state: RalphState) -> str:
    """Decide whether to loop back to write_code or finish."""
    passed = state.get("tests_passed", False)
    attempt = state["attempt_number"]
    if passed:
        return "end"
    if attempt >= MAX_ATTEMPTS:
        return "end"  # Failsafe: stop after max attempts even if not passing.
    return "retry"


# ---------------------------------------------------------------------------
# 6. BUILD THE GRAPH
# ---------------------------------------------------------------------------
# The graph is a state machine: START → write_code → run_tests → then either
# "retry" (back to write_code) or "end" (stop). conditional_edges means: after
# run_tests, call route_after_tests(state); if it returns "retry", go to
# write_code; if "end", go to END.

builder = StateGraph(RalphState)

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
# 7. RUN: Loop or Analyst demo
# ---------------------------------------------------------------------------


def run_analyst_demo() -> None:
    """Run the financial analyst chain (Ralph-Proof) on a few example inputs."""
    print(
        "Ralph-Proof Financial Analyst — structured extraction with graceful exit (qwen3)\n"
    )

    # Mix of finance-related and off-topic (e.g. cooking) to show NOT_FINANCE handling
    examples = [
        "U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, as Anthropic launched new artificial intelligence integrations aimed at enterprise customers, driving software subscription trends upward."
        "Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery’s studios and HBO Max is expected to close ahead of schedule in Q3 2026, shifting the landscape of streaming consolidation and media market caps.",
        "Crude oil traded higher this morning, breaking the $71 per barrel mark, amid rising tensions between the U.S. and Iran ahead of the highly anticipated nuclear talks in Geneva.",
        "CoStar Group (CSGP) posted full-year 2025 results showing 19% revenue growth and announced a massive $700 million share repurchase program for 2026, signaling strong capital return for investors.",
        "To make a traditional Beef Bourguignon, slowly braise a chuck roast in a full-bodied Pinot Noir with pearl onions, mushrooms, and thick-cut bacon. Let it simmer at 160°C for at least three hours until the meat is fork-tender.",
    ]

    for i, news in enumerate(examples, 1):
        print(f"--- Example {i} ---")
        print("News:", news[:60] + "..." if len(news) > 60 else news)
        result = run_analyst_chain(news)
        print("Result:", result)
        print()


def run_loop_demo() -> None:
    """Run the Ralph Wiggum iterative code-generation loop."""
    generated_path = OUTPUT_DIR / "generated.py"
    existing_code = ""
    initial_feedback = ""
    # If we already have a generated.py (e.g. from a previous run or failed run), load it
    # into initial state so the first write_code call can *enhance* it instead of starting from scratch.
    if generated_path.exists():
        existing_code = generated_path.read_text(encoding="utf-8").strip()
        initial_feedback = "A previous version of the script exists. Enhance it to fully meet the task (fix errors, ensure it runs)."
        print(
            "Ralph Wiggum Technique — found existing generated.py, will enhance it on retry\n"
        )
    else:
        print("Ralph Wiggum Technique — iterative loop with fresh context\n")

    # Initial state: task (long prompt), optional previous code, feedback, and counters.
    initial_state: RalphState = {
        "task_description": (
            "Write a function named analyze_news that takes a news article and returns a financial analysis. "
            "Follow the 'LANGCHAIN STRUCTURED OUTPUT' documentation in the system message: use ChatOllama and "
            "with_structured_output (do NOT use legacy langchain.chains, langchain.llms.Ollama, or PromptTemplateChain). "
            "Use a Pydantic class FinancialAnalysis with: sentiment (str: Bullish/Bearish/Neutral), impact_score (int 1-10), "
            "ticker_symbols (list), summary (str). Get the structured runnable with llm.with_structured_output(FinancialAnalysis) "
            "and invoke it; the result is already the Pydantic instance. "
            "If the input is NOT finance-related (e.g. cooking, sports, entertainment with no financial angle), "
            "return the exact string 'NOT_FINANCE' instead of forcing a financial analysis. "
            "The generated script must be heavily commented: docstrings for the module and all functions, section comments for logical blocks, and inline comments where they clarify non-obvious logic."
        ),
        "current_code": existing_code,
        "feedback": initial_feedback,
        "attempt_number": 1,
        "tests_passed": False,
    }
    print("Task:", initial_state["task_description"])
    print("Max attempts:", MAX_ATTEMPTS, "\n")

    # Run the graph until we hit END (success or max attempts). Each retry goes write_code → run_tests again.
    final_state = graph.invoke(initial_state)

    print("--- Final state ---")
    print("Attempts used:", final_state["attempt_number"])
    print("Feedback (last):", final_state["feedback"] or "(none)")
    print("\nGenerated code:\n")
    print(final_state["current_code"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "generated.py"
    out_file.write_text(final_state["current_code"], encoding="utf-8")
    print(f"\nResult written to: {out_file}")


def main() -> None:
    """Entry point: either run the code-generation loop or the analyst demo."""
    parser = argparse.ArgumentParser(description="Ralph Wiggum Technique demos")
    parser.add_argument(
        "--analyst",
        action="store_true",
        help="Run the financial analyst chain (Ralph-Proof extraction) instead of the loop",
    )
    args = parser.parse_args()

    if args.analyst:
        run_analyst_demo()
    else:
        run_loop_demo()


if __name__ == "__main__":
    main()
