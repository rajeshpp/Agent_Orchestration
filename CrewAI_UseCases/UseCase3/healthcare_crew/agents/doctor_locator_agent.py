from crewai import Agent
from connectors.maps_connector import find_doctors_near_location

class DoctorLocatorAgent(Agent):
    def locate(self, location, specialty=None):
        return find_doctors_near_location(location, specialty)

name = "Doctor Locator"
role = f"{name} Agent: Finds nearby hospitals, clinics, or specialists via Google Maps or hospital APIs."
doctor_locator_agent = DoctorLocatorAgent(
    name=name,
    role=role,
    goal="Recommend accessible local care options based on patient location and urgency.",
    backstory="A navigation agent integrated with Google Maps and hospital directories.",
    verbose=True
)
