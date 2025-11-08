from crewai import Agent

name = "Orchestrator"
role = f"{name} Agent: Coordinates and manages the healthcare triage process."
orchestrator_agent = Agent(
    name=name,
    role=role,
    goal="Orchestrate symptom analysis and provide a final structured report.",
    backstory="An experienced healthcare AI orchestrator.",
    verbose=True
)
