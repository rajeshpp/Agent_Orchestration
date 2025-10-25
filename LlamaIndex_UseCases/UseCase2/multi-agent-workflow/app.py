import os
import re
import json
import asyncio
from typing import Any, Dict, List, Tuple, Optional

from tabulate import tabulate
from llama_index.llms.openai import OpenAI

# =========================
# LLM Setup
# =========================
# Set your key in environment or replace below
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "<your_key_here>")
llm = OpenAI(model="gpt-4o-mini")  # or gpt-4o, etc.

# =========================
# Helper Utilities
# =========================

def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences if present."""
    text = text.strip()
    fence_pattern = r"^```(?:json)?\s*(.*?)\s*```$"
    m = re.match(fence_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text

def extract_json(text: str) -> Optional[Any]:
    """
    Try hard to parse JSON from LLM output.
    - Remove code fences
    - If still fails, grab substring between first '{' and last '}' and parse.
    """
    raw = strip_code_fences(text)
    try:
        return json.loads(raw)
    except Exception:
        # Try to locate the JSON object boundaries
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
        # Try array
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return None

def print_title(title: str):
    print("\n" + title)
    print("-" * len(title))

def print_kv_table(title: str, kv_pairs: List[Tuple[str, str]]):
    print_title(title)
    rows = [(k, v) for k, v in kv_pairs if v is not None]
    if rows:
        print(tabulate(rows, headers=["Field", "Value"], tablefmt="github"))
    else:
        print("(no data)")

def print_list_table(title: str, items: Optional[List[str]]):
    print_title(title)
    if items and isinstance(items, list):
        print(tabulate([[i] for i in items], headers=["Items"], tablefmt="github"))
    else:
        print("(no data)")

def print_meal_plan_table(title: str, plan: Optional[Dict[str, Dict[str, str]]]):
    print_title(title)
    if not plan or not isinstance(plan, dict):
        print("(no data)")
        return
    # Normalize to Day, Breakfast, Lunch, Dinner, Snacks
    headers = ["Day", "Breakfast", "Lunch", "Dinner", "Snacks"]
    rows = []
    for day in sorted(plan.keys(), key=lambda d: (d.startswith("Day "), int(re.sub(r"[^0-9]", "", d) or 0))):
        day_data = plan.get(day, {}) or {}
        rows.append([
            day,
            day_data.get("Breakfast", ""),
            day_data.get("Lunch", ""),
            day_data.get("Dinner", ""),
            day_data.get("Snacks", ""),
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))

# =========================
# Agent Functions (Manual Chain)
# =========================

async def diagnosis_agent(patient_input: str) -> str:
    prompt = f"""
You are a medical diagnosis agent.

Patient data:
{patient_input}

Diagnose for diabetes and related risk.

Return STRICT JSON with keys:
- condition (string)
- risk (string)
- recommended_tests (array of strings)
"""
    res = llm.complete(prompt)
    return res.text

async def treatment_agent(diagnosis_json_text: str) -> str:
    prompt = f"""
You are a diabetes treatment planning agent.

Use this diagnosis JSON:
{diagnosis_json_text}

Return STRICT JSON with keys:
- treatment_guidance (object) with keys: diet, exercise, weight_management, medication (object with first_line, additional_medications)
- monitoring_plan (object) with keys: blood_glucose, HbA1c, weight, blood_pressure
- doctor_recommendations (object) with keys: follow_up, education, screening
"""
    res = llm.complete(prompt)
    return res.text

async def diet_agent(diagnosis_json_text: str, treatment_json_text: str) -> str:
    prompt = f"""
You are a diabetic diet planning agent.

Inputs:
Diagnosis JSON:
{diagnosis_json_text}

Treatment JSON:
{treatment_json_text}

Return STRICT JSON with keys:
- foods_to_avoid (array of strings)
- foods_to_include (array of strings)
- 7_day_plan (object) where keys are Day 1..Day 7 and each value is an object with Breakfast, Lunch, Dinner, Snacks
"""
    res = llm.complete(prompt)
    return res.text

# =========================
# Main App
# =========================

async def main():
    print("✅ Diabetes Multi-Agent Workflow (Tabular) Ready")

    while True:
        patient_input = input("\nEnter patient data (or 'exit'): ").strip()
        if patient_input.lower() == "exit":
            break

        print("\n=============== RUNNING AGENT CHAIN ===============")

        # Run agents
        diagnosis_text = await diagnosis_agent(patient_input)
        treatment_text = await treatment_agent(diagnosis_text)
        diet_text = await diet_agent(diagnosis_text, treatment_text)

        # Parse JSONs
        diagnosis = extract_json(diagnosis_text) or {}
        treatment = extract_json(treatment_text) or {}
        diet = extract_json(diet_text) or {}

        # ---------- Diagnosis ----------
        cond = diagnosis.get("condition")
        risk = diagnosis.get("risk")
        tests = diagnosis.get("recommended_tests") if isinstance(diagnosis.get("recommended_tests"), list) else None

        print_kv_table("🩺 Diagnosis", [
            ("Condition", cond),
            ("Risk", risk),
        ])
        print_list_table("Recommended Tests", tests)

        # ---------- Treatment ----------
        tg = treatment.get("treatment_guidance") or {}
        med = tg.get("medication") or {}

        treat_rows = [
            ("Diet", tg.get("diet")),
            ("Exercise", tg.get("exercise")),
            ("Weight Management", tg.get("weight_management")),
            ("Medication - First Line", med.get("first_line") if isinstance(med, dict) else None),
            ("Medication - Additional", med.get("additional_medications") if isinstance(med, dict) else None),
        ]
        print_kv_table("💊 Treatment Guidance", treat_rows)

        mp = treatment.get("monitoring_plan") or {}
        print_kv_table("Monitoring Plan", [
            ("Blood Glucose", mp.get("blood_glucose")),
            ("HbA1c", mp.get("HbA1c")),
            ("Weight", mp.get("weight")),
            ("Blood Pressure", mp.get("blood_pressure")),
        ])

        dr = treatment.get("doctor_recommendations") or {}
        print_kv_table("Doctor Recommendations", [
            ("Follow-up", dr.get("follow_up")),
            ("Education", dr.get("education")),
            ("Screening", dr.get("screening")),
        ])

        # ---------- Diet ----------
        foods_avoid = diet.get("foods_to_avoid") if isinstance(diet.get("foods_to_avoid"), list) else None
        foods_include = diet.get("foods_to_include") if isinstance(diet.get("foods_to_include"), list) else None
        weekly_plan = diet.get("7_day_plan") if isinstance(diet.get("7_day_plan"), dict) else None

        print_list_table("🥗 Foods to Avoid", foods_avoid)
        print_list_table("✅ Foods to Include", foods_include)
        print_meal_plan_table("📅 7-Day Meal Plan", weekly_plan)

        print("\n==================== DONE ====================\n")


if __name__ == "__main__":
    asyncio.run(main())
