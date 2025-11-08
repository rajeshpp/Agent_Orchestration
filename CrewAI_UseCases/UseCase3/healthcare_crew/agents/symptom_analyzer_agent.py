from crewai import Agent

name = "Symptom Analyzer"
role = f"{name} Agent: Analyzes symptoms and identifies possible causes."
symptom_analyzer = Agent(
    name=name,
    role=role,
    goal="Suggest possible medical conditions for user's symptoms.",
    backstory="An expert medical AI trained on disease associations.",
    verbose=True
)
