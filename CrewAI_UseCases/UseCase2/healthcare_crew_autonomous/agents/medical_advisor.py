from typing import Dict, Any, List
from agents.base_agent import BaseAgent
import json

class MedicalAdvisor(BaseAgent):
    def __init__(self, db_path: str):
        super().__init__(name="Medical Advisor", role="Recommends next steps based on probable diagnoses")
        with open(db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)

    def receive(self, message: Dict[str, Any]) -> Dict[str, Any]:
        candidates = message.get("candidates", [])
        symptoms_text = message.get("symptoms_text", "")
        recommendations: List[str] = []
        urgent = False
        reasons = []

        for c in candidates:
            disease = self.db.get(c["id"])
            if not disease:
                continue
            if c.get("flags"):
                urgent = True
                reasons.append(f"Red flag(s) for {disease['name']}: {c.get('flags')}")
            if disease.get("severity") == "high" and c["score"] > 0.4:
                urgent = True
                reasons.append(f"High severity candidate: {disease['name']} (score {c['score']})")
            recs = disease.get("recommended_tests", []) + disease.get("advice", [])
            for r in recs:
                if r not in recommendations:
                    recommendations.append(r)
        if urgent:
            recommendations.insert(0, "URGENT: Seek immediate medical attention if symptoms are severe or worsening.")
        if not recommendations:
            recommendations.append("No clear match — monitor symptoms, rest, and seek primary care if persistent or worse.")

        summary = {
            "agent": self.name,
            "recommendations": recommendations,
            "reasons": reasons
        }
        return summary
