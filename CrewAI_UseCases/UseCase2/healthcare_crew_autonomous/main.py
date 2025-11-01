from crewai import Crew
from agents.orchestrator_agent import orchestrator_agent
from agents.symptom_analyzer_agent import symptom_analyzer
from agents.medical_advisor_agent import medical_advisor

from tasks.triage_patient import triage_patient_task
from tasks.analyze_symptoms import analyze_symptoms_task
from tasks.recommend_action import recommend_action_task

def get_agent_name(agent):
    # Try CrewAI Agent attributes/configs
    if hasattr(agent, "name"):
        return getattr(agent, "name")
    if hasattr(agent, "config") and agent.config is not None and "name" in agent.config:
        return agent.config["name"]
    return repr(agent)

def run_healthcare_crew(symptom_description: str):
    print("\n🩺 Running Healthcare Triage Crew...\n")
    agents = [orchestrator_agent, symptom_analyzer, medical_advisor]
    for agent in agents:
        print(f"Agent Started: {get_agent_name(agent)}")

    crew = Crew(
        agents=agents,
        tasks=[triage_patient_task, analyze_symptoms_task, recommend_action_task],
        verbose=True
        # If your CrewAI version supports it, try: tracing=True,
    )
    result = crew.kickoff(inputs={"symptoms": symptom_description})
    output = getattr(result, "output", result)

    print("\n=== HEALTHCARE TRIAGE SUMMARY ===\n")
    input_text = output['input_symptoms'] if 'input_symptoms' in output else symptom_description
    print(f"PATIENT PRESENTATION:\n  {input_text}\n")

    candidates = output['candidates'] if 'candidates' in output else []
    print("SYMPTOM ANALYZER - POSSIBLE CAUSES (with reasoning):")
    if candidates:
        for idx, cand in enumerate(candidates, 1):
            flags = ", ".join(cand.get('flags', [])) if cand.get('flags') else ""
            print(f" {idx}. {cand['name']} (score: {cand['score']})" + (f" | Flags: {flags}" if flags else ""))
    print("\nCLINICAL REASONING:")
    reasoning = output["analyzer_reasoning"] if "analyzer_reasoning" in output else ""
    print(f"  {reasoning}\n")

    recommendations = output['recommendations'] if 'recommendations' in output else []
    print("MEDICAL ADVISOR - ACTION PLAN:")
    for idx, rec in enumerate(recommendations, 1):
        print(f" {idx}. {rec}")

    rationale = output['advisor_reasons'] if 'advisor_reasons' in output else []
    if rationale:
        print("\nRATIONALE FOR RECOMMENDATIONS:")
        for idx, rr in enumerate(rationale, 1):
            print(f" {idx}. {rr}")

    # If tracing dashboard URL is available, print it
    dashboard_url = getattr(result, "trace_url", None)
    if dashboard_url:
        print(f"\nCrewAI Trace Dashboard: {dashboard_url}\n")

    print("\n--- End of Structured Healthcare Report ---\n")

if __name__ == "__main__":
    symptom_input = input("Describe your symptoms (e.g., 'I have fever, cough, and fatigue'): ")
    run_healthcare_crew(symptom_input)
