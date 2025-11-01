from crewai import Agent

symptom_analyzer = Agent(
    name="Symptom Analyzer",
    role="Analyzes symptoms and identifies possible causes.",
    goal="Suggest possible medical conditions for user's symptoms.",
    backstory="An expert medical AI trained on disease associations.",
    verbose=True
)
