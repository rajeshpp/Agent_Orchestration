from crewai import Task
from agents.orchestrator import orchestrator_agent

triage_patient_task = Task(
    description=(
        "Coordinate the overall triage workflow. "
        "Delegate symptom analysis to the Symptom Analyzer Agent and next-step recommendations "
        "to the Medical Advisor Agent. Combine the outputs into a single structured report."
    ),
    expected_output="A structured final triage summary with causes and recommendations.",
    agent=orchestrator_agent
)
