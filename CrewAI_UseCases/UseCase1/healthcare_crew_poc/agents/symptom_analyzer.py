from crewai import Agent

symptom_analyzer = Agent(
    name="Symptom Analyzer",
    role="Analyzes symptoms and identifies possible causes",
    goal="Understand symptoms and suggest possible medical conditions",
    backstory="A medical AI trained on symptoms and disease associations using clinical reasoning.",
    verbose=True
)
