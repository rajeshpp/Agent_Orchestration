from crewai import Crew

from agents.orchestrator import orchestrator_agent
from agents.symptom_analyzer import symptom_analyzer
from agents.medical_advisor import medical_advisor

from tasks.triage_patient import triage_patient_task
from tasks.analyze_symptoms import analyze_symptoms_task
from tasks.recommend_action import recommend_action_task

def run_healthcare_crew(symptom_description: str):
    print("\n🩺 Running Healthcare Triage Crew...\n")
    crew = Crew(
        agents=[orchestrator_agent, symptom_analyzer, medical_advisor],
        tasks=[triage_patient_task, analyze_symptoms_task, recommend_action_task],
        verbose=True
    )
    result = crew.kickoff(inputs={"symptoms": symptom_description})

    print("\n✅ Final Healthcare Report:\n", result)

if __name__ == "__main__":
    symptom_input = input("Describe your symptoms (e.g., 'I have fever, cough, and fatigue'): ")
    run_healthcare_crew(symptom_input)
