from crewai import Agent
from connectors.drug_api_connector import check_drug_interactions

class DrugInteractionCheckerAgent(Agent):
    def check(self, drugs):
        return check_drug_interactions(drugs)

name = "Drug Interaction Checker"
role = f"{name} Agent: Checks for possible drug-drug interactions using pharmacy APIs or clinical databases."
drug_interaction_checker_agent = DrugInteractionCheckerAgent(
    name=name,
    role=role,
    goal="Evaluate medication safety, flag risky combinations, warn of severe adverse events.",
    backstory="A clinical pharmacy agent leveraging up-to-date medical and drug references.",
    verbose=True
)
