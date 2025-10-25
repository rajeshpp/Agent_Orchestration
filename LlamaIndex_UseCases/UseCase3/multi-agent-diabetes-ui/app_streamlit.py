import os
import re
import json
import io
import datetime as dt
from typing import Any, Dict, Optional, List

import streamlit as st
from tabulate import tabulate
from dotenv import load_dotenv

# -----------------------------
# Optional LlamaIndex (RAG)
# -----------------------------
HAS_LLAMA = True
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core import Settings as LISettings
    from llama_index.llms.openai import OpenAI as LI_OpenAI
except Exception:
    HAS_LLAMA = False

# -----------------------------
# OpenAI (responses)
# -----------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # handled later

load_dotenv()

# =============== UI CONFIG ===============
st.set_page_config(
    page_title="Multi-Agent Diabetes AI (POC)",
    page_icon="🩺",
    layout="wide"
)

st.title("🎨 Multi-Agent Diabetes AI — POC (Diagnosis → Treatment → Diet)")
st.caption("Streamlit UI • Optional RAG via LlamaIndex • Manual agent chain for robustness")

# =============== SIDEBAR ===============
with st.sidebar:
    st.subheader("🔑 API & Model")
    api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-translate"])
    use_rag = st.checkbox("Use RAG (LlamaIndex) with ./data/*", value=True)
    st.write(f"LlamaIndex available: {'✅' if HAS_LLAMA else '❌'}")
    st.divider()
    st.subheader("📥 Upload extra .txt (optional)")
    extra_files = st.file_uploader("Add guidelines/notes (.txt)", type=["txt"], accept_multiple_files=True)
    st.caption("Uploaded files will be used for retrieval context in addition to ./data/* if RAG is enabled.")

# Guardrails
if not OpenAI:
    st.error("`openai` package missing. Install requirements and restart.")
    st.stop()
if not api_key:
    st.warning("Enter your OPENAI_API_KEY in the sidebar to run the agents.")
    st.stop()

# =============== CLIENT INIT ===============
oclient = OpenAI(api_key=api_key)

# =============== RAG (Optional) ===============
retriever = None
if use_rag and HAS_LLAMA:
    # Configure LlamaIndex to use OpenAI via your key (for embeddings/LLM if needed)
    LISettings.llm = LI_OpenAI(model=model_name, api_key=api_key)

    # Build a temporary in-memory directory with base ./data + uploads
    # We’ll load data/ plus any uploaded text files
    os.makedirs("data", exist_ok=True)
    uploaded_temp_paths = []
    if extra_files:
        for f in extra_files:
            path = os.path.join("data", f"{dt.datetime.now().strftime('%Y%m%d%H%M%S')}_{f.name}")
            with open(path, "wb") as out:
                out.write(f.read())
            uploaded_temp_paths.append(path)

    try:
        docs = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(docs)
        retriever = index.as_retriever(similarity_top_k=3)
    except Exception as e:
        st.warning(f"RAG disabled (build error): {e}")
        retriever = None

# =============== HELPERS ===============
def strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text

def extract_json(text: str) -> Optional[Any]:
    raw = strip_code_fences(text)
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end+1])
            except Exception:
                pass
        start = raw.find("["); end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end+1])
            except Exception:
                pass
    return None

def pretty_table_from_json(obj: Any, title: str) -> str:
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                v_disp = json.dumps(v, indent=2, ensure_ascii=False)
            else:
                v_disp = str(v)
            rows.append((k, v_disp))
        return f"### {title}\n\n" + tabulate(rows, headers=["Field", "Value"], tablefmt="github")
    elif isinstance(obj, list):
        rows = [[json.dumps(x, ensure_ascii=False)] for x in obj]
        return f"### {title}\n\n" + tabulate(rows, headers=["Items"], tablefmt="github")
    return f"### {title}\n\n(no data)"

def call_openai_json(prompt: str) -> str:
    """
    Ask model to return STRICT JSON. We’ll still parse defensively.
    """
    sys = (
        "You are a clinical assistant. "
        "Always return STRICT JSON only; do not include explanations or code fences."
    )
    resp = oclient.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ]
    )
    return resp.choices[0].message.content.strip()

def build_context(query: str) -> str:
    if not retriever:
        return ""
    try:
        nodes = retriever.retrieve(query)
        chunks = [n.get_content() for n in nodes]
        ctx = "\n\n".join(chunks[:5])
        return ctx
    except Exception:
        return ""

# =============== AGENTS (Manual Chain) ===============
def diagnosis_agent(patient_input: str, ctx: str) -> str:
    prompt = f"""
Patient data:
{patient_input}

Context (optional):
{ctx}

Return STRICT JSON with keys:
- condition (string)
- risk (string)
- rationale (string)
- recommended_tests (array of strings)
"""
    return call_openai_json(prompt)

def treatment_agent(diagnosis_json: str, ctx: str) -> str:
    prompt = f"""
You are a treatment planning agent.

Diagnosis JSON:
{diagnosis_json}

Context (optional):
{ctx}

Return STRICT JSON with keys:
- treatment_guidance (object) with keys:
    - diet (string)
    - exercise (string)
    - weight_management (string)
    - medication (object with first_line, additional_medications)
- monitoring_plan (object) with keys:
    - blood_glucose (string)
    - HbA1c (string)
    - weight (string)
    - blood_pressure (string)
- doctor_recommendations (object) with keys:
    - follow_up (string)
    - education (string)
    - screening (string)
"""
    return call_openai_json(prompt)

def diet_agent(diagnosis_json: str, treatment_json: str, ctx: str) -> str:
    prompt = f"""
You are a diabetic diet planning agent.

Diagnosis JSON:
{diagnosis_json}

Treatment JSON:
{treatment_json}

Context (optional):
{ctx}

Return STRICT JSON with keys:
- foods_to_avoid (array of strings)
- foods_to_include (array of strings)
- 7_day_plan (object) where keys are "Day 1".."Day 7"
  and each day has Breakfast, Lunch, Dinner, Snacks (strings)
"""
    return call_openai_json(prompt)

# =============== MAIN UI ===============
colL, colR = st.columns([1.2, 1])
with colL:
    st.subheader("🧾 Patient Input")
    default_example = "HbA1c: 7.4, BMI 31, thirsty, fatigue"
    patient_text = st.text_area("Enter symptoms/metrics", value=default_example, height=120)

    run = st.button("▶️ Run Multi-Agent Workflow", type="primary")

with colR:
    st.subheader("⚙️ Options")
    export_format = st.selectbox("Export report format", ["Markdown", "JSON"])

st.divider()

tab_diag, tab_treat, tab_diet, tab_all = st.tabs(["🩺 Diagnosis", "💊 Treatment", "🥗 Diet", "📄 Full Report"])

if run:
    with st.status("Running agents…", expanded=False) as status:
        # RAG context per step (can differ slightly)
        ctx_diag = build_context(patient_text) if use_rag else ""
        diag_raw = diagnosis_agent(patient_text, ctx_diag)
        diag_json = extract_json(diag_raw) or {"raw": diag_raw}

        ctx_treat = build_context("diabetes treatment guidelines") if use_rag else ""
        treat_raw = treatment_agent(json.dumps(diag_json), ctx_treat)
        treat_json = extract_json(treat_raw) or {"raw": treat_raw}

        ctx_diet = build_context("diabetes diet low GI high fiber") if use_rag else ""
        diet_raw = diet_agent(json.dumps(diag_json), json.dumps(treat_json), ctx_diet)
        diet_json = extract_json(diet_raw) or {"raw": diet_raw}

        status.update(label="Completed ✅", state="complete", expanded=False)

    # ---------- Render Diagnosis ----------
    with tab_diag:
        st.markdown(pretty_table_from_json({
            "condition": diag_json.get("condition"),
            "risk": diag_json.get("risk"),
            "rationale": diag_json.get("rationale"),
            "recommended_tests": diag_json.get("recommended_tests"),
        }, "Diagnosis"))

    # ---------- Render Treatment ----------
    with tab_treat:
        st.markdown(pretty_table_from_json(treat_json.get("treatment_guidance"), "Treatment Guidance"))
        st.markdown(pretty_table_from_json(treat_json.get("monitoring_plan"), "Monitoring Plan"))
        st.markdown(pretty_table_from_json(treat_json.get("doctor_recommendations"), "Doctor Recommendations"))

    # ---------- Render Diet ----------
    with tab_diet:
        st.markdown(pretty_table_from_json(diet_json.get("foods_to_avoid"), "Foods to Avoid"))
        st.markdown(pretty_table_from_json(diet_json.get("foods_to_include"), "Foods to Include"))
        st.markdown(pretty_table_from_json(diet_json.get("7_day_plan"), "7-Day Meal Plan"))

    # ---------- Full Report & Export ----------
    with tab_all:
        report = {
            "timestamp": dt.datetime.now().isoformat(),
            "patient_input": patient_text,
            "diagnosis": diag_json,
            "treatment": treat_json,
            "diet": diet_json,
            "rag_used": bool(retriever),
        }

        if export_format == "JSON":
            st.code(json.dumps(report, indent=2), language="json")
            buf = io.BytesIO(json.dumps(report, indent=2).encode("utf-8"))
            st.download_button("⬇️ Download JSON", buf, file_name="diabetes_multi_agent_report.json", mime="application/json")
        else:
            md_parts = [
                f"# Multi-Agent Diabetes Report — {report['timestamp']}",
                "## Patient Input",
                f"{patient_text}",
                "## Diagnosis",
                "```json",
                json.dumps(diag_json, indent=2),
                "```",
                "## Treatment",
                "```json",
                json.dumps(treat_json, indent=2),
                "```",
                "## Diet",
                "```json",
                json.dumps(diet_json, indent=2),
                "```",
                f"**RAG used:** {'Yes' if retriever else 'No'}"
            ]
            md = "\n\n".join(md_parts)
            st.markdown(md)
            buf = io.BytesIO(md.encode("utf-8"))
            st.download_button("⬇️ Download Markdown", buf, file_name="diabetes_multi_agent_report.md", mime="text/markdown")

else:
    st.info("Enter patient data and click **Run** to generate the multi-agent report.")
