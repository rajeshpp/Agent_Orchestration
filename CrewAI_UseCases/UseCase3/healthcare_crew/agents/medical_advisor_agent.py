from crewai import Agent

name = "Medical Advisor"
role = f"{name} Agent: Recommends next steps based on analyzed symptoms."
medical_advisor = Agent(
    name=name,
    role=role,
    goal="Offer advice, tests, and escalation if needed.",
    backstory="A responsible virtual healthcare adviser.",
    verbose=True
)
