from crewai import Task
from agents.drug_interaction_checker_agent import drug_interaction_checker_agent

def check_drug_interaction_task_fn(inputs):
    drugs = inputs.get("medications", [])
    return {
        "drug_interactions": drug_interaction_checker_agent.check(drugs)
    }

check_drug_interaction_task = Task(
    description="Check for dangerous interactions among patient medications and triage recommendations.",
    expected_output="List of flagged medications/interactions and safety advice.",
    agent=drug_interaction_checker_agent,
    run=check_drug_interaction_task_fn
)
