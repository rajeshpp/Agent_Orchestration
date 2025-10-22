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