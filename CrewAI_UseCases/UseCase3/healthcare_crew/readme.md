# 🧠 CrewAI Healthcare Autonomous Multi-Agent System  

### 🩺 An End-to-End Autonomous Healthcare Triage & Advisory Platform  

Built using **CrewAI**, **Ollama**, and **Azure OpenAI**, this project demonstrates how specialized AI agents can collaborate, reason, and act autonomously to deliver real-world healthcare intelligence — from symptom analysis to doctor discovery.

---

## 🚀 Overview  

This project brings **Agentic AI** to life in the healthcare domain.  
Each agent has a distinct responsibility — from fetching patient history to detecting drug interactions — and all communicate through **CrewAI’s orchestration layer**.  

The result: an **autonomous healthcare ecosystem** capable of real reasoning and real action.  


## ⚙️ Tech Stack

🧩 CrewAI – Multi-agent orchestration

🧠 LLM Backend: Ollama (local) / Azure OpenAI (cloud)

📚 Data: JSON-based knowledge & drug info

🗺️ APIs: Google Maps / Hospital API integration

🐍 Language: Python 3.10+

## 📁 Project Structure
```
/healthcare-crewai
│
├── agents/
│   ├── orchestrator_agent.py
│   ├── symptom_analyzer_agent.py
│   ├── knowledge_base_agent.py
│   ├── medical_advisor_agent.py
│   ├── patient_history_agent.py
│   ├── drug_interaction_agent.py
│   ├── doctor_locator_agent.py
│   └── reasoning_engine.py
│
├── data/
│   ├── diseases.json
│   └── drug_data.json
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧩 Agent Ecosystem  

| # | Agent | Purpose | Example Action |
|---|--------|----------|----------------|
| 1️⃣ | **OrchestratorAgent** | Central coordinator | Routes tasks and merges outputs |
| 2️⃣ | **SymptomAnalyzerAgent** | Diagnoses | Maps symptoms to likely conditions |
| 3️⃣ | **KnowledgeBaseAgent** | Retrieves structured data | Pulls info from `diseases.json` |
| 4️⃣ | **MedicalAdvisorAgent** | Suggests next steps | Gives clinical recommendations |
| 5️⃣ | **PatientHistoryFetcherAgent** | Fetches past records | Queries patient database |
| 6️⃣ | **DrugInteractionCheckerAgent** | Ensures medication safety | Detects harmful combinations |
| 7️⃣ | **DoctorLocatorAgent** | Connects to care | Finds nearby specialists via Google Maps or hospital APIs |
| 8️⃣ | **ReasoningEngine** | Synthesizes insights | Builds structured triage reports |

---

## 🧠 Example Flow  

**User Input:**  
> “I have chest pain and dizziness after taking a new medication.”

**Autonomous Flow:**  
1️⃣ Patient History Fetcher → retrieves cardiac record  
2️⃣ Symptom Analyzer → identifies Angina or side effect  
3️⃣ Knowledge Base → loads relevant disease data  
4️⃣ Drug Interaction Checker → detects risky drug combo  
5️⃣ Medical Advisor → provides next steps  
6️⃣ Doctor Locator → finds nearest cardiologist  
7️⃣ Reasoning Engine → generates structured triage summary  
8️⃣ Orchestrator → finalizes the output  

**Final Output Example:**  
```json
{
  "condition": "Possible drug-induced chest pain (interaction risk ⚠️)",
  "recommendation": "Stop medication immediately and consult Dr. Mehta (2.1 km away)",
  "urgency": "High",
  "confidence": 0.91
}
