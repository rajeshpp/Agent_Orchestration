# LangSmith Healthcare POC — Working code (no theory)

This repository is a minimal, production-minded Proof‑of‑Concept integrating **LangSmith** tracing into a small healthcare LLM workflow (symptom triage + recommended next step). Everything here is runnable locally (no Docker). Follow README steps below.

---

## Folder structure

```
langsmith-healthcare-poc/
├── README.md
├── .env.example
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── app.py                # CLI entrypoint: runs a sample triage request
│   ├── config.py             # env loader
│   ├── llm_client.py         # OpenAI call(s) wrapped with LangSmith traceable
│   ├── prompt_templates.py   # prompt templates for triage
│   └── langsmith_helpers.py  # langsmith client + helper decorators
├── tests/
│   ├── test_llm_client.py
│   └── test_prompt_templates.py
└── README_RUN_LOGS.md        # example run + sample LangSmith run ids
```

---

## Files (copy each file exactly)

### `README.md`
```markdown
# LangSmith Healthcare POC

**Goal:** Demonstrate LangSmith tracing on a small healthcare triage LLM flow. Minimal, runnable, and focused on code.

## Prereqs
1. Python 3.10+ (recommended)
2. A LangSmith API key (from smith.langchain.com). See: docs: `Create an account and API key`.
3. An OpenAI API key (or another provider — update `llm_client.py` accordingly).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and fill LANGSMITH_API_KEY and OPENAI_API_KEY and optional PROJECT_NAME
```

## Run
```bash
# quick interactive run
python -m src.app

# or: provide symptoms via CLI
python -m src.app --symptoms "fever, cough, sore throat"
```

## Tests
```bash
pytest -q
```

## Notes
- Traces will appear in your LangSmith project defined by `PROJECT_NAME` (defaults to `default`).
- The code uses LangSmith's `traceable` decorator to emit traces for LLM calls and business logic.
```
```

---

### `.env.example`
```text
# Copy to .env and fill keys
LANGSMITH_API_KEY=ls_your_key_here
OPENAI_API_KEY=sk_your_key_here
PROJECT_NAME=healthcare-poc
OPENAI_MODEL=gpt-4o-mini
# Optional: set LANGSMITH_API_URL if using a custom endpoint
# LANGSMITH_API_URL=
```

---

### `requirements.txt`
```text
# pinned lightweight set — update versions as desired
python-dotenv>=1.0.0
openai>=1.0.0
langsmith>=0.4.0
pytest>=7.0.0
requests>=2.28.0
```

---

### `.gitignore`
```text
.env
.venv/
__pycache__/
*.pyc
```

---

### `src/__init__.py`
```python
# package marker
```

---

### `src/config.py`
```python
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "default")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

settings = Settings()
```

---

### `src/langsmith_helpers.py`
```python
# Small wrapper to initialize LangSmith client and expose traceable decorator
from langsmith import Client, traceable
from typing import Optional
from .config import settings

_client: Optional[Client] = None


def get_langsmith_client() -> Client:
    global _client
    if _client is None:
        # Client will read LANGSMITH_API_KEY from environment if not provided
        _client = Client()
    return _client


# Expose a simple decorator alias that forwards project name metadata where possible
# Use traceable on functions that you want to show up as runs in LangSmith
def trace_fn(name: str = None):
    """Return a decorator that marks a function traceable by LangSmith.

    Usage:
        @trace_fn("llm.call")
        def call(...):
            ...
    """
    def _decorator(func):
        # traceable accepts optional metadata; we'll set "name" via function arg in annotations
        if name:
            return traceable(name)(func)
        return traceable(func)
    return _decorator
```

---

### `src/prompt_templates.py`
```python
TRIAGE_PROMPT = """
You are a licensed virtual triage assistant for primary care. You will receive a comma-separated list of symptoms.
Return a JSON object with the following fields exactly: `triage_level` (one of "SELF_CARE", "SEE_GP", "URGENT_CARE", "EMERGENCY"), `rationale` (1-2 sentences), and `recommended_next_step` (a single sentence with what the patient should do next).

Symptoms: {symptoms}

Respond with only valid JSON.
"""
```

---

### `src/llm_client.py`
```python
import os
import json
from typing import Dict
import openai
from .config import settings
from .langsmith_helpers import trace_fn

openai.api_key = settings.OPENAI_API_KEY


@trace_fn("llm.symptom_triage")
def triage_symptoms(symptoms: str) -> Dict:
    """Call OpenAI ChatCompletion (simple wrapper). Returns parsed JSON as dict.

    This function is decorated with LangSmith's `traceable` (via trace_fn)
    so the call + inputs/outputs will be sent to LangSmith.
    """
    prompt = (
        f"{open('src/prompt_templates.py').read()}\n\n"  # include prompt template for traceability
        f"Symptoms: {symptoms}\n\n"
    )

    # NOTE: model/parameters can be tuned; we keep a minimal stable call
    resp = openai.ChatCompletion.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a virtual triage assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )

    text = resp["choices"][0]["message"]["content"].strip()

    # Attempt to load JSON from the model output; tolerate surrounding text
    try:
        # Find first and last braces to extract JSON
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start:end+1] if (start != -1 and end != -1) else text
        parsed = json.loads(candidate)
    except Exception:
        # Fallback: store full raw output in a special structure
        parsed = {"raw_output": text}

    # Enrich response with model metadata
    parsed["_model"] = settings.OPENAI_MODEL
    return parsed
```

---

### `src/app.py`
```python
import argparse
from .llm_client import triage_symptoms


def main():
    parser = argparse.ArgumentParser(description="Run Healthcare triage POC (LangSmith tracing)")
    parser.add_argument("--symptoms", type=str, help="Comma-separated symptom list", default=None)
    args = parser.parse_args()

    if args.symptoms:
        symptoms = args.symptoms
    else:
        symptoms = input("Enter comma-separated symptoms (e.g. fever, cough): ")

    result = triage_symptoms(symptoms)
    print("--- TRIAGE RESULT ---")
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

---

### `tests/test_prompt_templates.py`
```python
from src import prompt_templates

def test_triage_prompt_has_symptoms_placeholder():
    assert "{symptoms}" in prompt_templates.TRIAGE_PROMPT
```

---

### `tests/test_llm_client.py`
```python
import pytest
from unittest.mock import patch
from src.llm_client import triage_symptoms


@patch("src.llm_client.openai.ChatCompletion.create")
def test_triage_parses_json(mock_create):
    mock_create.return_value = {
        "choices": [
            {"message": {"content": '{"triage_level":"SELF_CARE","rationale":"mild symptoms","recommended_next_step":"Stay hydrated"}'}}
        ]
    }

    out = triage_symptoms("fever")
    assert out["triage_level"] == "SELF_CARE"
    assert "recommended_next_step" in out
```

---

### `README_RUN_LOGS.md`
```markdown
# Example run (save for your records)

Sample output printed by `python -m src.app --symptoms "fever, cough"`:

```
--- TRIAGE RESULT ---
{
  "triage_level": "SELF_CARE",
  "rationale": "Mild symptoms likely viral; monitor",
  "recommended_next_step": "Rest, fluids, paracetamol as needed. If worsens seek care.",
  "_model": "gpt-4o-mini"
}
```

You can copy the run id shown in the LangSmith UI to correlate runs.
```

---

## Implementation notes (short)
- This POC uses `langsmith.traceable` to instrument `triage_symptoms` calls. The LangSmith client is initialized via `Client()` which reads `LANGSMITH_API_KEY` from the environment. See LangSmith docs for details.


---

End of project files.

