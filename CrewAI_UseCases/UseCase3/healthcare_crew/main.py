from crewai import Crew
from agents.orchestrator_agent import orchestrator_agent
from agents.symptom_analyzer_agent import symptom_analyzer
from agents.medical_advisor_agent import medical_advisor
from agents.patient_history_fetcher_agent import patient_history_fetcher_agent
from agents.drug_interaction_checker_agent import drug_interaction_checker_agent
from agents.doctor_locator_agent import doctor_locator_agent

from tasks.triage_patient import triage_patient_task
from tasks.analyze_symptoms import analyze_symptoms_task
from tasks.recommend_action import recommend_action_task
from tasks.fetch_patient_history import fetch_patient_history_task
from tasks.check_drug_interaction import check_drug_interaction_task
from tasks.locate_doctor import locate_doctor_task

def get_agent_name(agent):
    if hasattr(agent, "name"):
        return getattr(agent, "name")
    if hasattr(agent, "config") and agent.config is not None and "name" in agent.config:
        return agent.config["name"]
    return repr(agent)

def run_healthcare_crew(symptom_description: str, patient_id: str, location: str):
    print("\n🩺 Running Healthcare Triage Crew...\n")
    agents = [
        orchestrator_agent,
        symptom_analyzer,
        medical_advisor,
        patient_history_fetcher_agent,
        drug_interaction_checker_agent,
        doctor_locator_agent
    ]
    for agent in agents:
        print(f"Agent Started: {get_agent_name(agent)}")

    crew = Crew(
        agents=agents,
        tasks=[
            fetch_patient_history_task,
            triage_patient_task,
            analyze_symptoms_task,
            check_drug_interaction_task,
            locate_doctor_task,
            recommend_action_task
        ],
        verbose=True
    )
    result = crew.kickoff(inputs={
        "symptoms": symptom_description,
        "patient_id": patient_id,
        "location": location
    })
    output = getattr(result, "output", result)

    print("\n=== HEALTHCARE TRIAGE SUMMARY ===\n")
    input_text = output['input_symptoms'] if 'input_symptoms' in output else symptom_description
    print(f"PATIENT PRESENTATION:\n  {input_text}\n")

    history = output['patient_history'] if 'patient_history' in output else {}
    if history:
        print("PATIENT HISTORY:")
        print(history)

    candidates = output['candidates'] if 'candidates' in output else []
    print("SYMPTOM ANALYZER - POSSIBLE CAUSES (with reasoning):")
    if candidates:
        for idx, cand in enumerate(candidates, 1):
            flags = ", ".join(cand.get('flags', [])) if cand.get('flags') else ""
            print(f" {idx}. {cand['name']} (score: {cand['score']})" + (f" | Flags: {flags}" if flags else ""))
    print("\nCLINICAL REASONING:")
    reasoning = output['analyzer_reasoning'] if 'analyzer_reasoning' in output else ""
    print(f"  {reasoning}\n")

    interactions = output['drug_interactions'] if 'drug_interactions' in output else []
    if interactions:
        print("DRUG INTERACTIONS:")
        for interaction in interactions:
            print(interaction)

    facilities = output['nearby_doctors'] if 'nearby_doctors' in output else []
    if facilities:
        print("DOCTORS & FACILITIES NEARBY:")
        for f in facilities:
            print(f)

    recommendations = output['recommendations'] if 'recommendations' in output else []
    print("MEDICAL ADVISOR - ACTION PLAN:")
    for idx, rec in enumerate(recommendations, 1):
        print(f" {idx}. {rec}")

    rationale = output['advisor_reasons'] if 'advisor_reasons' in output else []
    if rationale:
        print("\nRATIONALE FOR RECOMMENDATIONS:")
        for idx, rr in enumerate(rationale, 1):
            print(f" {idx}. {rr}")

    dashboard_url = getattr(result, "trace_url", None)
    if dashboard_url:
        print(f"\nCrewAI Trace Dashboard: {dashboard_url}\n")

    print("\n--- End of Structured Healthcare Report ---\n")

if __name__ == "__main__":
    symptom_input = input("Describe your symptoms (e.g., 'I have fever, cough, and fatigue'): ")
    patient_id = input("Patient ID: ")
    location = input("Location: ")
    run_healthcare_crew(symptom_input, patient_id, location)
