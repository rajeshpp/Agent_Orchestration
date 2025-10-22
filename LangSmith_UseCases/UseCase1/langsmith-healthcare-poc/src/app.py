import argparse
from .llm_client import triage_symptoms


def main():
    parser = argparse.ArgumentParser(description="Run Healthcare triage POC (LangSmith tracing)")
    parser.add_argument("--symptoms", type=str, help="Comma-separated symptom list", default=None)
    args = parser.parse_args()

    if args.symptoms:
        symptoms = args.symptoms
    else:
        symptoms = input("Enter comma-separated symptoms (e.g. fever, cough): ")

    result = triage_symptoms(symptoms)
    print("--- TRIAGE RESULT ---")
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
