from crewai import Task
from agents.doctor_locator_agent import doctor_locator_agent

def locate_doctor_task_fn(inputs):
    location = inputs.get("location")
    specialty = inputs.get("specialty", None)
    return {
        "nearby_doctors": doctor_locator_agent.locate(location, specialty)
    }

locate_doctor_task = Task(
    description="Locate nearby hospitals, clinics, or doctors based on symptoms and patient location.",
    expected_output="List of recommended medical facilities and contact info.",
    agent=doctor_locator_agent,
    run=locate_doctor_task_fn
)
