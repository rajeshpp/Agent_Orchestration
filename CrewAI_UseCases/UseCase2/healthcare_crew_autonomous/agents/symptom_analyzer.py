import json
from typing import Dict, Any, List, Tuple
from agents.base_agent import BaseAgent

def load_db(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def tokenize(text: str) -> List[str]:
    return [t.strip().lower() for t in text.replace("/", " ").replace(",", " ").split() if t.strip()]

def score_disease(symptom_tokens: List[str], disease: Dict) -> Tuple[float, List[str]]:
    disease_tokens = []
    for s in disease.get("symptoms", []):
        disease_tokens += tokenize(s)
    disease_tokens = set(disease_tokens)
    overlap = sum(1 for t in symptom_tokens if t in disease_tokens)
    base_score = overlap / max(1, len(disease_tokens))
    flagged = []
    for rf in disease.get("red_flags", []):
        for t in tokenize(rf):
            if t in symptom_tokens:
                flagged.append(rf)
                base_score += 0.35 # flag boost
                break
    return min(base_score, 1.0), flagged

class SymptomAnalyzer(BaseAgent):
    def __init__(self, db_path: str):
        super().__init__(name="Symptom Analyzer", role="Identifies possible medical conditions from symptoms")
        self.db = load_db(db_path)

    def receive(self, message: Dict[str, Any]) -> Dict[str, Any]:
        symptoms_text = message.get("symptoms_text", "")
        tokens = tokenize(symptoms_text)
        scores = []
        for key, disease in self.db.items():
            sc, flags = score_disease(tokens, disease)
            if sc > 0:
                scores.append({
                    "id": key,
                    "name": disease.get("name"),
                    "score": round(sc, 3),
                    "flags": flags,
                    "matched_tokens": [t for t in tokens if t in " ".join(disease.get("symptoms", [])).lower()]
                })
        scores.sort(key=lambda x: x["score"], reverse=True)
        reasoning = f"Analyzed {len(self.db)} diseases; found {len(scores)} candidates."
        return {"agent": self.name, "candidates": scores[:5], "reasoning": reasoning}
