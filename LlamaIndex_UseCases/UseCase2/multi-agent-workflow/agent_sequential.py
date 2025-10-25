import os
import json
import re

# Try to import the OpenAI client; prefer the modern `openai` package
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


def extract_json_from_text(text: str):
    if not text or not isinstance(text, str):
        return None
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    try:
                        cleaned = re.sub(r",\s*}\s*$", "}", candidate)
                        return json.loads(cleaned)
                    except Exception:
                        return None
    return None


def call_openai(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in environment to run agent_sequential.py")

    # Try the official openai package ChatCompletion
    if OPENAI_AVAILABLE:
        try:
            # Some installations use openai.ChatCompletion.create
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            # choose text from choices
            if resp and hasattr(resp, "choices"):
                text = ""
                for c in resp.choices:
                    # older libs put content in c.message.content or c.text
                    if getattr(c, "message", None) and getattr(c.message, "content", None):
                        text += c.message.content
                    elif getattr(c, "text", None):
                        text += c.text
                return text.strip()
        except Exception:
            # Fall back to HTTP-like response via openai.ChatCompletion
            try:
                return resp.choices[0].message.content
            except Exception:
                pass

    # If we reach here we couldn't call openai package; try environment-based HTTP call
    raise RuntimeError("OpenAI client not available or call failed. Install `openai` Python package.")


def run_sequence(patient_input: str, model: str = "gpt-4o-mini"):
    """Run the three agents sequentially via separate OpenAI calls.

    Returns a dict with keys: diagnosis_resp, diag_json, treatment_resp, treat_json, diet_resp, diet_json
    """
    # System prompts (same contract as in app.py)
    diagnosis_prompt = (
        "You are DiagnosisAgent.\n"
        "Input: patient_profile (user provided).\n"
        "Output: Return ONLY a JSON object matching this schema:\n"
        "{\n  \"condition_likelihood\": \"normal|prediabetes|diabetes\",\n  \"risk\": \"low|medium|high\",\n  \"recommended_tests\": [\"test1\", \"test2\"],\n  \"notes\": \"brief clinical notes\",\n  \"summary\": \"short human summary\"\n}\n"
        "Do not include extra explanation outside the JSON."
    )

    treatment_prompt_template = (
        "You are TreatmentAgent.\n"
        "Input: diagnosis JSON.\n"
        "Output: Return ONLY a JSON object with this schema:\n"
        "{\n  \"lifestyle_guidance\": \"short text\",\n  \"suggested_medication\": [\"med1 (consult doctor)\"],\n  \"monitoring_frequency\": \"e.g. monthly\",\n  \"notes\": \"brief notes\",\n  \"summary\": \"short human summary\"\n}\n"
        "Do not include extra explanation outside the JSON."
    )

    diet_prompt_template = (
        "You are DietAgent.\n"
        "Input: diagnosis JSON and treatment JSON.\n"
        "Output: Provide a detailed human-readable dietary plan. It's OK to return free text. If possible include a short JSON under a top-level 'diet_json' key.\n"
    )

    out = {
        "diagnosis_resp": None,
        "diag_json": None,
        "treatment_resp": None,
        "treat_json": None,
        "diet_resp": None,
        "diet_json": None,
    }

    # Run DiagnosisAgent
    diagnosis_resp = call_openai(diagnosis_prompt, patient_input, model=model)
    out["diagnosis_resp"] = diagnosis_resp
    out["diag_json"] = extract_json_from_text(diagnosis_resp)

    # Run TreatmentAgent
    treatment_input = diagnosis_resp if out["diag_json"] is None else json.dumps(out["diag_json"])
    treatment_resp = call_openai(treatment_prompt_template, treatment_input, model=model)
    out["treatment_resp"] = treatment_resp
    out["treat_json"] = extract_json_from_text(treatment_resp)

    # Run DietAgent
    diet_input = "Patient profile:\n" + patient_input + "\n\nDiagnosis:\n" + (json.dumps(out["diag_json"]) if out["diag_json"] else diagnosis_resp) + "\n\nTreatment:\n" + (json.dumps(out["treat_json"]) if out["treat_json"] else treatment_resp)
    diet_resp = call_openai(diet_prompt_template, diet_input, model=model)
    out["diet_resp"] = diet_resp
    out["diet_json"] = extract_json_from_text(diet_resp)

    return out


def main():
    print("Sequential Agent Runner — Diagnosis -> Treatment -> Diet")
    patient_input = input("Enter patient data (eg: HbA1c: 7.4, BMI 31, fatigue): ")
    try:
        out = run_sequence(patient_input)
        print("\n--- DiagnosisAgent raw output:\n", out["diagnosis_resp"])
        print("\n--- DiagnosisAgent parsed JSON:\n", json.dumps(out["diag_json"], indent=2) if out["diag_json"] else "(none)")
        print("\n--- TreatmentAgent raw output:\n", out["treatment_resp"])
        print("\n--- TreatmentAgent parsed JSON:\n", json.dumps(out["treat_json"], indent=2) if out["treat_json"] else "(none)")
        print("\n--- DietAgent raw output:\n", out["diet_resp"])
        print("\n--- DietAgent parsed JSON:\n", json.dumps(out["diet_json"], indent=2) if out["diet_json"] else "(none)")
    except Exception as e:
        print("Error while calling OpenAI:", e)


if __name__ == '__main__':
    main()
