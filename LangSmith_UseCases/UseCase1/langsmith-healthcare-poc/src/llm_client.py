import json
from typing import Dict
from openai import OpenAI
from .config import settings
from .langsmith_helpers import trace_fn

# Create OpenAI client (new API interface)
client = OpenAI(api_key=settings.OPENAI_API_KEY)


@trace_fn(run_type="llm", name="triage_symptoms")
def triage_symptoms(symptoms: str) -> Dict:
    """
    Call OpenAI ChatCompletion using v2 API.
    Returns parsed JSON as dict.
    """
    # Explicitly import template instead of reading file
    from .prompt_templates import TRIAGE_PROMPT
    prompt = TRIAGE_PROMPT.format(symptoms=symptoms)

    # The new API uses client.chat.completions.create()
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a virtual triage assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )

    text = response.choices[0].message.content.strip()

    try:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start:end + 1] if (start != -1 and end != -1) else text
        parsed = json.loads(candidate)
    except Exception:
        parsed = {"raw_output": text}

    parsed["_model"] = settings.OPENAI_MODEL
    return parsed
