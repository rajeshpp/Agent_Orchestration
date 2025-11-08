from crewai import Agent
from connectors.db_connector import fetch_patient_history

class PatientHistoryFetcherAgent(Agent):
    def fetch(self, patient_id):
        return fetch_patient_history(patient_id)

name = "Patient History Fetcher"
role = f"{name} Agent: Fetches patient's medical history from the hospital/clinic database."
patient_history_fetcher_agent = PatientHistoryFetcherAgent(
    name=name,
    role=role,
    goal="Retrieve relevant patient information (diagnoses, medications, allergies) for triage.",
    backstory="A secure medical agent that accesses local EMR and hospital databases rapidly as needed.",
    verbose=True
)
