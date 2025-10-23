import os
from .graph_builder import build_graph

# LangSmith tracing environment setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "brain-tumor-risk"

# ✅ Exported graph instance (Studio looks for this)
app = build_graph()

def main():
    """Run the brain tumor risk workflow locally."""
    runner = app.compile()

    patient_data = {
        "age": 52,
        "gender": "Male",
        "symptoms": "Persistent headaches, blurred vision",
        "mri_report": "Slight hyperintensity in temporal lobe region",
    }

    result = runner.invoke({"patient_data": patient_data})
    print("--- FINAL REPORT ---")
    print(result["final_report"])

if __name__ == "__main__":
    main()
