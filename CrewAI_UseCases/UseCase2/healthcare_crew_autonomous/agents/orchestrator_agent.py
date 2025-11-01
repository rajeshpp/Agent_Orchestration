from crewai import Agent

orchestrator_agent = Agent(
    name="Orchestrator Agent",
    role="Coordinates and manages the healthcare triage process",
    goal="Orchestrate symptom analysis and provide a final structured report.",
    backstory="An experienced healthcare AI orchestrator.",
    verbose=True
)
