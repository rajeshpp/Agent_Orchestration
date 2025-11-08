from crewai import Task
from agents.medical_advisor_agent import medical_advisor

recommend_action_task = Task(
    description="Recommend next steps based on the identified causes — including lifestyle tips or when to see a doctor.",
    expected_output="Clear next-step plan for the user.",
    agent=medical_advisor
)
