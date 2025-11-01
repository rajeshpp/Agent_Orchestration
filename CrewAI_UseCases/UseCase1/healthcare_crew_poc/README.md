# 🩺 Healthcare Symptom Triage CrewAI POC

### Description
A CrewAI-based MVP that simulates a healthcare triage assistant with two collaborating agents.

### Agents
1. **Symptom Analyzer** – Infers possible causes.
2. **Medical Advisor** – Suggests next steps or tests.
3. **Orchestrator Agent** - Coordinates workflow between agents

### Run
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=.....
python main.py
```