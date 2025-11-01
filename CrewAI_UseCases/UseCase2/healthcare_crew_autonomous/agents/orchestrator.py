from typing import Dict, Any
from agents.base_agent import BaseAgent
from agents.symptom_analyzer import SymptomAnalyzer
from agents.medical_advisor import MedicalAdvisor

class Orchestrator(BaseAgent):
    def __init__(self, db_path: str):
        super().__init__(name="Orchestrator", role="Coordinates the triage workflow")
        self.symptom_analyzer = SymptomAnalyzer(db_path)
        self.medical_advisor = MedicalAdvisor(db_path)

    def receive(self, message: Dict[str, Any]) -> Dict[str, Any]:
        symptoms_text = message.get("symptoms_text", "")

        print("[Orchestrator] -> Symptom Analyzer")
        analyze_resp = self.symptom_analyzer.receive({"symptoms_text": symptoms_text})

        print("[Orchestrator] -> Medical Advisor")
        advisor_resp = self.medical_advisor.receive({
            "candidates": analyze_resp.get("candidates", []),
            "symptoms_text": symptoms_text
        })

        triage_report = {
            "input_symptoms": symptoms_text,
            "candidates": analyze_resp.get("candidates", []),
            "analyzer_reasoning": analyze_resp.get("reasoning"),
            "recommendations": advisor_resp.get("recommendations"),
            "advisor_reasons": advisor_resp.get("reasons")
        }

        return {"agent": self.name, "triage_report": triage_report}
