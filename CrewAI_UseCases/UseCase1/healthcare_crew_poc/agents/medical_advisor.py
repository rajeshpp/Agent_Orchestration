from crewai import Agent

medical_advisor = Agent(
    name="Medical Advisor",
    role="Advises next steps based on analyzed symptoms",
    goal="Recommend next steps — rest, OTC medicines, tests, or doctor visit.",
    backstory="An AI assistant that provides health recommendations based on medical best practices.",
    verbose=True
)
