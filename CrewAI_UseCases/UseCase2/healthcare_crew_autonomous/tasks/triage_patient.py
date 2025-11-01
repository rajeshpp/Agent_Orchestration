from crewai import Task
from agents.orchestrator_agent import orchestrator_agent

triage_patient_task = Task(
    description="Coordinate the overall triage workflow. Delegate symptom analysis and recommendation.",
    expected_output="A structured final triage summary with causes and recommendations.",
    agent=orchestrator_agent
)
