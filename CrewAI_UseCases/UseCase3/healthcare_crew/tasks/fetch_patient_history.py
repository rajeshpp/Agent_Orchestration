from crewai import Task
from agents.patient_history_fetcher_agent import patient_history_fetcher_agent

def fetch_history_task(inputs):
    patient_id = inputs.get("patient_id")
    return {
        "patient_history": patient_history_fetcher_agent.fetch(patient_id)
    }

fetch_patient_history_task = Task(
    description="Fetch the patient's medical history and records from the database for context.",
    expected_output="Summary of relevant history, diagnoses, allergies and medications.",
    agent=patient_history_fetcher_agent,
    run=fetch_history_task
)
