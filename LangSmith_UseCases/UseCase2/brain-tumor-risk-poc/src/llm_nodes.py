from openai import OpenAI
from langsmith import traceable
from .config import settings
from pathlib import Path

client = OpenAI(api_key=settings.OPENAI_API_KEY)

@traceable(run_type="chain", name="preprocess_node")
def preprocess_node(inputs):
    # Normalize patient data
    data = inputs.get("patient_data", {})
    text = " ".join(f"{k}:{v}" for k, v in data.items())
    return {"normalized_text": text}


@traceable(run_type="llm", name="analysis_node")
def analysis_node(inputs):
    text = inputs.get("normalized_text")
    template = Path("src/prompts/analyze_prompt.txt").read_text()
    prompt = template.replace("{patient_info}", text)
    
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"risk_summary": resp.choices[0].message.content}



@traceable(run_type="llm", name="explanation_node")
def explanation_node(inputs):
    summary = inputs.get("risk_summary")
    prompt = Path("src/prompts/explain_prompt.txt").read_text().format(summary=summary)
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"final_report": resp.choices[0].message.content}
