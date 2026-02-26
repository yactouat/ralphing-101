#!/usr/bin/env python3
"""
Ralph Wiggum Technique — Autonomous AI development loop with fresh context.

This script demonstrates an iterative while-loop where an AI coding agent is
repeatedly given a task until it succeeds or hits the attempt cap. Each iteration:
  1. Reads PRD.md to understand the task specification.
  2. Loads progress.txt (running summary + last failure reason) as context.
  3. Creates a FRESH LLM instance (no shared state between iterations).
  4. Generates or fixes Python code for the analyze_news() function.
  5. Runs tests against 10 examples (7 finance, 3 non-finance).
  6. Writes results and failure reasons into progress.txt for the next iteration.

The key insight: each iteration sees only the PRD, the progress log, and the
previous script — never a growing conversation. This avoids "context rot".

Requirements: Python 3.12+, langchain-ollama, Ollama (llama3.1).

How to run (use the project venv):
  source venv/bin/activate
  python ralph_wiggum_technique.py              # runs the code-generation loop
  python ralph_wiggum_technique.py --analyst   # runs the analyst on example inputs
"""

# Standard library
import argparse
import importlib.util
import sys
from pathlib import Path

# LangChain: local LLM (Ollama) and message types
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 10           # Cap iterations so the loop never runs forever.
OLLAMA_MODEL = "llama3.1"
OUTPUT_DIR = Path("out/ralph_wiggum_technique")
PRD_PATH = Path("PRD.md")   # Task specification read at startup.
PROGRESS_FILE = OUTPUT_DIR / "progress.txt"  # Persistent scratchpad across iterations.

# Injected into every prompt so the LLM uses current LangChain APIs, not legacy ones.
DOCS = """
MODERN LANGCHAIN REFERENCE

AVAILABLE IMPORTS — use exactly these:
  from langchain_ollama import ChatOllama
  from langchain_core.messages import HumanMessage, SystemMessage
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.output_parsers import StrOutputParser
  from pydantic import BaseModel, Field
  from typing import Literal

Use Pydantic (pydantic.BaseModel) for ALL schema-related operations — it is the
standard for structured output, data validation, and type enforcement in this stack.

--- 1. STRUCTURED OUTPUT (Pydantic) ---

Use Pydantic BaseModel to define the schema. Pass it to with_structured_output() so
the LLM returns a validated, typed object.

  from langchain_ollama import ChatOllama
  from langchain_core.prompts import ChatPromptTemplate
  from pydantic import BaseModel, Field
  from typing import Literal

  class BookInfo(BaseModel):
      title: str = Field(..., description="Book title")
      year: int  = Field(..., description="Publication year")
      # Use Literal for enums — Pydantic enforces the allowed values automatically.
      genre: Literal["fiction", "non-fiction", "poetry"] = Field(..., description="Genre")

  llm = ChatOllama(model="llama3.1", temperature=0.2)
  structured_llm = llm.with_structured_output(BookInfo)
  result = structured_llm.invoke("Tell me about Dune.")
  # result is a BookInfo instance — access fields as attributes.
  print(result.title, result.year, result.genre)

  # Pydantic also handles validation: raise ValueError for invalid input.
  # Use model_validate() to validate a plain dict against the schema:
  try:
      obj = BookInfo.model_validate({"title": "Dune", "year": 1965, "genre": "fiction"})
  except Exception as exc:
      raise ValueError(f"Invalid data: {exc}")  # Pydantic raises ValidationError on bad data

--- 1b. INVOKE INPUT TYPES ---

What you pass to .invoke() depends on what you're calling it on:

  - llm or structured_llm (no prompt template): pass a str or list of BaseMessages
  - prompt | llm  /  prompt | structured_llm: pass a dict of template variables

  from langchain_core.messages import HumanMessage, SystemMessage

  # Invoking structured_llm directly — use a str or list of BaseMessages:
  structured_llm = llm.with_structured_output(MyModel)
  result = structured_llm.invoke(text)                          # str
  result = structured_llm.invoke([HumanMessage(content=text)]) # messages list

  # Invoking a chain that starts with a prompt template — use a dict:
  chain = prompt | structured_llm
  result = chain.invoke({"article": text})   # dict matches template variables

--- 2. PARSING CHAIN RESPONSES ---

The return type of chain.invoke() depends on what the chain ends with:

  a) chain ends with the LLM (prompt | llm)
     → returns an AIMessage object; read the text via .content (a str).

  b) chain ends with StrOutputParser (prompt | llm | StrOutputParser())
     → returns a plain str directly; no .content needed.

  c) chain uses with_structured_output(MyModel) (prompt | structured_llm)
     → returns a Pydantic model instance; access fields as attributes.

Examples:

  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.output_parsers import StrOutputParser
  from langchain_ollama import ChatOllama
  from pydantic import BaseModel, Field
  from typing import Literal

  llm = ChatOllama(model="llama3.1", temperature=0.2)
  prompt = ChatPromptTemplate.from_messages([
      ("system", "You are a helpful assistant."),
      ("human", "{text}"),
  ])

  # Case (a): raw AIMessage — use .content to get the string
  chain_a = prompt | llm
  msg = chain_a.invoke({"text": "Say hello."})
  text: str = msg.content          # ← always use .content here

  # Case (b): parsed to str — no .content, just use the value directly
  chain_b = prompt | llm | StrOutputParser()
  text: str = chain_b.invoke({"text": "Say hello."})   # already a str

  # Case (c): structured output — Pydantic instance, access via attributes
  class Sentiment(BaseModel):
      label: Literal["positive", "negative", "neutral"] = Field(..., description="Sentiment label")
      score: float = Field(..., description="Confidence score between 0 and 1")

  structured_llm = llm.with_structured_output(Sentiment)
  chain_c = prompt | structured_llm
  result = chain_c.invoke({"text": "Markets are up today."})
  print(result.label, result.score)   # ← attribute access

--- 3. PROMPT STRINGS IN ChatPromptTemplate ---

ChatPromptTemplate treats every {word} as a template variable placeholder.
Describe the output schema in plain English inside the prompt.
When using with_structured_output(), the Pydantic Field descriptions already
communicate the schema to the LLM — the prompt needs only the input variable.

  prompt = ChatPromptTemplate.from_messages([
      ("system", "Classify the text. Return a label (str) and a confidence score (float 0-1)."),
      ("human", "{text}"),
  ])
  chain = prompt | llm.with_structured_output(MyModel)
  chain.invoke({"text": "Some input"})

--- 4. A SINGLE CHAIN (prompt | model) ---

Use the pipe operator to compose a prompt template with a model:

  from langchain_core.prompts import ChatPromptTemplate
  from langchain_ollama import ChatOllama

  llm = ChatOllama(model="llama3.1", temperature=0.3)
  prompt = ChatPromptTemplate.from_messages([
      ("system", "You are a helpful assistant."),
      ("human", "Translate '{text}' to {language}."),
  ])
  chain = prompt | llm
  response = chain.invoke({"text": "Hello, world!", "language": "Spanish"})
  print(response.content)   # AIMessage → read via .content → "¡Hola, mundo!"

--- 5. MULTI-STEP CHAIN (logical check then action) ---

Chains can encode decisions, not just transformations.  A first step
classifies the input; a second step acts only when the first step gives
the right answer.  Classify first, then branch — reject off-topic input
early rather than passing it blindly to downstream processing.

  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.output_parsers import StrOutputParser
  from langchain_ollama import ChatOllama

  llm = ChatOllama(model="llama3.1", temperature=0.1)
  parser = StrOutputParser()

  # Step 1 — logical gate: is the input a question?
  #   StrOutputParser turns the AIMessage into a plain str automatically.
  classify_prompt = ChatPromptTemplate.from_messages([
      ("system", "Reply with only 'yes' or 'no'."),
      ("human", "Is the following text a question?\n\n{text}"),
  ])
  classify_chain = classify_prompt | llm | parser   # output: plain str "yes" | "no"

  # Step 2 — conditional action: answer only when classification says yes.
  answer_prompt = ChatPromptTemplate.from_messages([
      ("human", "Answer the following question in one sentence:\n\n{text}"),
  ])
  answer_chain = answer_prompt | llm | parser   # output: plain str

  def route(verdict: str, original_text: str) -> str:
      if verdict.strip().lower().startswith("yes"):
          return answer_chain.invoke({"text": original_text})   # returns str
      raise ValueError("Input is not a question — cannot answer.")

  text = "What is the boiling point of water at sea level?"
  verdict = classify_chain.invoke({"text": text})   # plain str, no .content needed
  result = route(verdict, text)
  print(result)   # e.g. "Water boils at 100°C (212°F) at sea level."
"""


# ---------------------------------------------------------------------------
# 2. TEST SUITE — 10 examples (7 finance, 3 non-finance)
# ---------------------------------------------------------------------------
# Each entry: (article_text, is_finance)
#   is_finance=True  → analyze_news() must return a structured object with the required keys
#   is_finance=False → analyze_news() must raise an exception (off-topic noise)

TEST_EXAMPLES = [
    # --- Finance (7) ---
    (
        "U.S. stocks closed higher on Tuesday, boosted by positive labor market reports, "
        "as the S&P 500 gained 1.2% and the Nasdaq rose 1.5%.",
        True,
    ),
    (
        "Netflix's planned $82.7 billion acquisition of Warner Bros. Discovery's studios "
        "and HBO Max is expected to close ahead of schedule in Q3 2026, shifting the "
        "landscape of streaming consolidation and media market caps.",
        True,
    ),
    (
        "Crude oil traded higher this morning, breaking the $71 per barrel mark, amid "
        "rising tensions between the U.S. and Iran ahead of nuclear talks in Geneva.",
        True,
    ),
    (
        "CoStar Group (CSGP) posted full-year 2025 results showing 19% revenue growth "
        "and announced a massive $700 million share repurchase program for 2026, "
        "signaling strong capital return for investors.",
        True,
    ),
    (
        "Apple (AAPL) reported record Q4 2025 earnings, with revenue up 12% year-over-year "
        "to $124 billion and EPS of $2.18, beating analyst estimates by 8 cents.",
        True,
    ),
    (
        "The Federal Reserve signaled a potential 25 basis point rate cut in Q2 2026 "
        "following lower-than-expected CPI data, sending Treasury yields lower and "
        "pushing equity markets to new highs.",
        True,
    ),
    (
        "Amazon (AMZN) announced a $2.5 billion acquisition of a leading AI healthcare "
        "startup, expanding its AWS cloud and enterprise health-services division.",
        True,
    ),
    # --- Non-finance (3) ---
    (
        "To make a traditional Beef Bourguignon, slowly braise a chuck roast in a "
        "full-bodied Pinot Noir with pearl onions, mushrooms, and thick-cut bacon. "
        "Let it simmer at 160°C for at least three hours until the meat is fork-tender.",
        False,
    ),
    (
        "The Golden State Warriors defeated the Boston Celtics 112-98 last night in a "
        "thrilling NBA playoff game, with Stephen Curry scoring 38 points and 9 assists.",
        False,
    ),
    (
        "Christopher Nolan's latest sci-fi epic received a standing ovation at the "
        "Cannes Film Festival, with critics praising its stunning practical effects "
        "and emotionally resonant screenplay.",
        False,
    ),
]


# ---------------------------------------------------------------------------
# 3. PROGRESS LOG HELPERS
# ---------------------------------------------------------------------------


def read_progress() -> str:
    """Return the contents of progress.txt, or an empty string if it doesn't exist yet."""
    if PROGRESS_FILE.exists():
        return PROGRESS_FILE.read_text(encoding="utf-8").strip()
    return ""


def generate_failure_analysis(prd_content: str, script_code: str, test_results: str) -> str:
    """
    Ask a fresh LLM instance to explain why the generated script failed the tests.

    Returns a short bullet-point analysis the next iteration can act on.
    Returns an empty string on any LLM error so a failure here never crashes the loop.
    """
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
    messages = [
        SystemMessage(content="You are a Python code reviewer. Be concise and precise."),
        HumanMessage(content=(
            "A Python script was supposed to implement analyze_news() per these requirements:\n"
            f"{prd_content}\n\n"
            "The script that was tested:\n"
            f"```python\n{script_code}\n```\n\n"
            "Test results:\n"
            f"{test_results}\n\n"
            "In 3-5 bullet points, identify the root causes of the failures "
            "and what must change in the next attempt."
        )),
    ]
    try:
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        return f"(failure analysis unavailable: {exc})"


def generate_progress_summary(prd_content: str, existing_progress: str) -> str:
    """
    Ask a fresh LLM instance to distil the existing progress log into a concise summary.

    Placed at the top of progress.txt so the next iteration sees a compact overview
    of everything tried so far before reading the latest attempt's full details.
    Returns an empty string when there is no prior history to summarise.
    """
    if not existing_progress:
        return ""

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
    messages = [
        SystemMessage(content="You are a software project manager. Be concise."),
        HumanMessage(content=(
            "Task requirements:\n"
            f"{prd_content}\n\n"
            "Development progress log so far:\n"
            f"{existing_progress}\n\n"
            "Write a concise summary (5-8 bullet points) covering:\n"
            "- What approaches have been tried\n"
            "- What has worked or partially worked\n"
            "- What keeps failing and why\n"
            "- Key constraints and lessons learned so far"
        )),
    ]
    try:
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        return f"(summary unavailable: {exc})"


def append_progress(
    attempt: int,
    passed: bool,
    test_results: str,
    script_code: str,
    failure_analysis: str,
    summary: str,
) -> None:
    """
    Rewrite progress.txt with a cumulative summary followed by the latest attempt's details.

    File structure after each call:
      === SUMMARY OF ATTEMPTS 1–{N-1} ===
      {LLM-generated summary of everything before this attempt}

      --- Attempt {N} — PASSED/FAILED ---
      SCRIPT TESTED:
      {full source of generated.py}

      TEST RESULTS:
      {pass/fail details from run_tests()}

      FAILURE ANALYSIS:
      {LLM explanation of root causes, or "(not applicable)" if passed}
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status = "PASSED" if passed else "FAILED"

    parts: list[str] = []

    # --- Cumulative summary of all prior attempts (empty on first run) ---
    if summary:
        parts.append(
            f"=== SUMMARY OF ATTEMPTS 1–{attempt - 1} ===\n"
            f"{summary}"
        )

    # --- Full details of the current attempt ---
    analysis_section = failure_analysis if not passed else "(not applicable — tests passed)"
    parts.append(
        f"--- Attempt {attempt} — {status} ---\n\n"
        f"SCRIPT TESTED:\n"
        f"```python\n{script_code}\n```\n\n"
        f"TEST RESULTS:\n{test_results}\n\n"
        f"FAILURE ANALYSIS:\n{analysis_section}"
    )

    PROGRESS_FILE.write_text("\n\n".join(parts).strip(), encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. CODE GENERATION — fresh LLM instance each call
# ---------------------------------------------------------------------------


def generate_code(prd_content: str, progress: str, attempt: int, existing_code: str) -> str:
    """
    Ask the LLM to write or fix analyze_news() based on the PRD and progress log.

    A brand-new ChatOllama instance is created on every call so there is no
    shared in-memory state between iterations (fresh context, no context rot).

    Args:
        prd_content:   Full text of PRD.md — defines the expected behaviour.
        progress:      Contents of progress.txt — history of what failed and why.
        attempt:       Current attempt number (1-based), shown in the prompt.
        existing_code: Previous generated.py, if any, so the model can fix it
                       rather than start from scratch.

    Returns:
        A string containing executable Python code (no markdown fences).
    """
    # --- Fresh LLM instance: no shared history across iterations ---
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

    # Build up the user prompt in logical sections
    prompt_parts: list[str] = []

    # Section 1 — task spec from PRD
    prompt_parts.append(f"PRODUCT REQUIREMENTS (from PRD.md):\n{prd_content}\n")

    # Section 2 — progress log (empty on first attempt)
    if progress:
        prompt_parts.append(
            "PROGRESS LOG (all previous attempts, failure reasons, and context):\n"
            f"{progress}\n"
        )

    # Section 3 — existing script to fix/enhance (if available)
    if existing_code:
        prompt_parts.append(
            "CURRENT SCRIPT (fix or enhance it — output the full improved script):\n"
            "```python\n"
            f"{existing_code}\n"
            "```\n"
        )
        prompt_parts.append(
            f"This is attempt {attempt}. Study the progress log to understand what "
            "failed and why, then output the complete fixed Python script. "
            "No markdown fences, no explanation."
        )
    else:
        prompt_parts.append(
            f"This is attempt {attempt}. Write the Python script from scratch. "
            "No markdown code fences, no explanation."
        )

    user_content = "\n".join(prompt_parts)

    system_content = (
        "You are a Python programmer. Write a script that contains an analyze_news(article) "
        "function matching the product requirements. "
        "The script must be heavily commented: module docstring, function docstrings, "
        "section comments, and inline comments where they clarify non-obvious logic. "
        "Reply with only executable Python code — no markdown, no prose.\n\n"
        + DOCS
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    raw: str = response.content if hasattr(response, "content") else str(response)

    # Strip markdown code fences if the LLM wrapped the output anyway
    code = raw.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return code


# ---------------------------------------------------------------------------
# 5. TEST RUNNER
# ---------------------------------------------------------------------------


def _call_analyze_news(generated_path: Path, article: str):
    """
    Load generated.py as a fresh module and call analyze_news(article).

    A fresh module load on every call avoids Python's module cache returning
    a stale version of generated.py from a previous iteration.

    Raises on any load or runtime error (caller catches and formats feedback).
    """
    spec = importlib.util.spec_from_file_location("generated", generated_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("importlib could not create a module spec for generated.py.")

    module = importlib.util.module_from_spec(spec)
    # Overwrite any cached version so we always execute the latest file
    sys.modules["generated"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    analyze_news = getattr(module, "analyze_news", None)
    if analyze_news is None or not callable(analyze_news):
        raise RuntimeError("Generated code has no callable 'analyze_news' function.")

    return analyze_news(article)


def run_tests(generated_path: Path) -> tuple[bool, str]:
    """
    Run all TEST_EXAMPLES against the generated analyze_news() function.

    For finance examples (is_finance=True):
      - Result must have 'sentiment' and 'impact_score' attributes (structured output object).

    For non-finance examples (is_finance=False):
      - analyze_news() must raise an exception (per PRD: off-topic noise raises an exception).

    Returns:
        (True, "All 10 tests passed.") if every example passes.
        (False, "<N> failures with descriptions") otherwise.
    """
    if not generated_path.exists():
        return False, "generated.py does not exist yet."

    failures: list[str] = []

    for i, (article, is_finance) in enumerate(TEST_EXAMPLES, 1):
        label = "finance" if is_finance else "non-finance"
        short = article[:80] + "..." if len(article) > 80 else article

        try:
            result = _call_analyze_news(generated_path, article)
        except Exception as exc:
            if not is_finance:
                # Correct behaviour: non-finance content must raise an exception.
                continue
            failures.append(
                f"Example {i} ({label}): EXCEPTION — {type(exc).__name__}: {exc}\n"
                f"  Article: {short}"
            )
            continue

        if not is_finance:
            # analyze_news() returned normally instead of raising — that's a failure.
            failures.append(
                f"Example {i} ({label}): expected an exception for non-finance content "
                f"but got a return value ({type(result).__name__}: {result!r}).\n"
                f"  Article: {short}"
            )
            continue

        # Finance example: result must be a structured output object.
        is_structured = hasattr(result, "sentiment") and hasattr(result, "impact_score")
        if not is_structured:
            failures.append(
                f"Example {i} ({label}): result is not a structured output object "
                f"(got {type(result).__name__}: {result!r}).\n"
                f"  Article: {short}"
            )

    if not failures:
        return True, f"All {len(TEST_EXAMPLES)} tests passed."

    summary = (
        f"{len(failures)}/{len(TEST_EXAMPLES)} tests failed:\n"
        + "\n".join(f"  [{j+1}] {f}" for j, f in enumerate(failures))
    )
    return False, summary


# ---------------------------------------------------------------------------
# 6. ANALYST DEMO — load generated.py and run it on example inputs
# ---------------------------------------------------------------------------


def run_analyst_chain(news_input: str):
    """
    Load out/ralph_wiggum_technique/generated.py and call analyze_news(news_input).

    Returns the result, or a graceful error message if the file is missing or
    the function raises (Ralph-style: no ugly tracebacks for the audience).
    """
    generated_path = OUTPUT_DIR / "generated.py"
    if not generated_path.exists():
        return (
            "Ralph says: 'I'm a financial analyst!' "
            "(Error: no generated.py found — run the loop first.)"
        )
    try:
        return _call_analyze_news(generated_path, news_input)
    except Exception as exc:
        return f"[Ralph Technique Triggered]: {type(exc).__name__}: {exc}"


def run_analyst_demo() -> None:
    """Run the financial analyst on a selection of example inputs and print the results."""
    print("Ralph-Proof Financial Analyst — structured extraction (llama3.1)\n")

    examples = [
        # Finance examples (distinct from the test suite)
        "Bitcoin surged past $95,000 on Monday as institutional demand accelerated "
        "following ETF inflows of over $1.2 billion in a single trading session.",
        "JPMorgan Chase reported Q1 2026 net income of $14.2 billion, up 9% year-over-year, "
        "driven by strong investment banking fees and net interest income growth.",
        "Arm Holdings priced its secondary offering of 75 million shares at $142 each, "
        "raising $10.65 billion in one of the largest tech follow-on offerings in history.",
        "Nvidia reported quarterly revenue of $38 billion, up 122% year-over-year, as demand "
        "for its Blackwell GPU architecture continued to outpace supply across data-center customers.",
        "The European Central Bank cut its benchmark rate by 25 basis points to 2.25%, citing "
        "easing inflation and slowing eurozone growth, its fourth consecutive reduction since late 2025.",
        "Walmart raised its full-year 2026 earnings guidance after posting a 6% same-store sales "
        "increase, crediting its expanded private-label grocery line and same-day delivery network.",
        "Pfizer announced a $14 billion deal to acquire a biotech firm specialising in RNA therapies, "
        "boosting its oncology pipeline and sending its share price up 4.7% in pre-market trading.",
        # Non-finance examples (distinct from the test suite)
        "NASA's Artemis IV crew completed a six-hour moonwalk near the lunar south pole, "
        "collecting rock samples expected to shed light on the Moon's volcanic history.",
        "Taylor Swift's 'Midnight Horizons' album debuted at number one in 47 countries, "
        "breaking the streaming record with 320 million plays in its first 24 hours.",
        "Spain clinched the 2026 FIFA World Cup title with a 2-1 victory over Brazil in the "
        "final played in New York, with Yamal scoring the decisive goal in extra time.",
    ]

    for i, news in enumerate(examples, 1):
        print(f"--- Example {i} ---")
        print("News:", news[:80] + "..." if len(news) > 80 else news)
        print("Result:", run_analyst_chain(news))
        print()


# ---------------------------------------------------------------------------
# 7. MAIN LOOP
# ---------------------------------------------------------------------------


def run_loop_demo() -> None:
    """
    Main while-loop: generate code → test it → update progress → repeat.

    Termination conditions:
      - All 10 tests pass   → success.
      - attempt > MAX_ATTEMPTS → give up and report failure.
    """
    # --- Load task spec from PRD.md ---
    if not PRD_PATH.exists():
        print(f"Error: {PRD_PATH} not found. Cannot determine task spec.")
        sys.exit(1)
    prd_content = PRD_PATH.read_text(encoding="utf-8").strip()

    # Count test split for display
    n_finance = sum(1 for _, f in TEST_EXAMPLES if f)
    n_other = len(TEST_EXAMPLES) - n_finance

    print("Ralph Wiggum Technique — while-loop, fresh LLM each iteration\n")
    print(f"Task spec   : {PRD_PATH}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Progress log: {PROGRESS_FILE}")
    print(f"Max attempts: {MAX_ATTEMPTS}")
    print(f"Test suite  : {len(TEST_EXAMPLES)} examples ({n_finance} finance, {n_other} non-finance)\n")

    generated_path = OUTPUT_DIR / "generated.py"
    attempt = 1

    while attempt <= MAX_ATTEMPTS:
        print(f"=== Attempt {attempt}/{MAX_ATTEMPTS} ===")

        # Load the running history of what happened and why things failed
        progress = read_progress()

        # Load the previous script from disk so the model can fix it rather
        # than regenerate from scratch (avoids throwing away prior good work)
        existing_code = ""
        if generated_path.exists():
            existing_code = generated_path.read_text(encoding="utf-8").strip()

        # Generate code — fresh LLM instance, no shared in-memory state
        print("Generating code (fresh LLM instance)...")
        code = generate_code(prd_content, progress, attempt, existing_code)

        # Persist the generated code
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        generated_path.write_text(code, encoding="utf-8")
        print(f"Code written → {generated_path}")

        # Run the full test suite
        print(f"Running {len(TEST_EXAMPLES)} tests...")
        passed, test_results = run_tests(generated_path)

        # Ask the LLM to explain why it failed (skipped when tests pass)
        failure_analysis = ""
        if not passed:
            print("Generating failure analysis (fresh LLM instance)...")
            failure_analysis = generate_failure_analysis(prd_content, code, test_results)

        # Summarise all prior history so the next iteration sees a compact overview
        print("Generating progress summary (fresh LLM instance)...")
        summary = generate_progress_summary(prd_content, progress)

        # Rewrite progress.txt: summary of prior attempts + full details of this attempt
        append_progress(attempt, passed, test_results, code, failure_analysis, summary)

        print(f"Result: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            # Print a short excerpt so the operator can follow along
            lines = test_results.splitlines()
            preview = "\n".join(lines[:6]) + ("\n  ..." if len(lines) > 6 else "")
            print(f"Feedback:\n{preview}\n")

        # Termination signal: all tests green
        if passed:
            print(f"\nAll {len(TEST_EXAMPLES)} tests passed on attempt {attempt}.")
            break

        attempt += 1

    else:
        # while-loop exhausted without break
        print(f"\nMax attempts ({MAX_ATTEMPTS}) reached. Last feedback in {PROGRESS_FILE}.")

    print(f"\nProgress log : {PROGRESS_FILE}")
    print(f"Generated code: {generated_path}")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: either run the code-generation loop or the analyst demo."""
    parser = argparse.ArgumentParser(description="Ralph Wiggum Technique demos")
    parser.add_argument(
        "--analyst",
        action="store_true",
        help="Run the financial analyst on example inputs instead of the generation loop",
    )
    args = parser.parse_args()

    if args.analyst:
        run_analyst_demo()
    else:
        run_loop_demo()


if __name__ == "__main__":
    main()
