from crewai import Task
from agents.symptom_analyzer_agent import symptom_analyzer

analyze_symptoms_task = Task(
    description="Analyze the user's symptoms and identify possible causes.",
    expected_output="List of 2-3 possible medical conditions with confidence and reasoning.",
    agent=symptom_analyzer
)
