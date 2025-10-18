# langflow_poc.py
"""
LangFlow POC:
 - Fetch basic example flows
 - Create/import one example into your workspace
 - Run it via /api/v1/run/{flow_id}
Requirements: requests
Env:
 - LANGFLOW_URL (default: http://127.0.0.1:7860)
 - LANGFLOW_API_KEY (from LangFlow UI)
"""

import os
import requests
import json
import time
import dotenv
dotenv.load_dotenv()

LANGFLOW_URL = os.environ.get("LANGFLOW_URL", "http://127.0.0.1:7860")
API_KEY = os.environ.get("LANGFLOW_API_KEY")
if not API_KEY:
    raise SystemExit("Set LANGFLOW_API_KEY env var (create one in LangFlow UI).")

HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def list_basic_examples():
    url = f"{LANGFLOW_URL}/api/v1/flows/basic_examples/"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

def create_flow_from_example(example_flow_json, new_name=None):
    # Example JSON from basic_examples is often just the full flow data.
    # We POST to /api/v1/flows/ to create the flow in the workspace.
    url = f"{LANGFLOW_URL}/api/v1/flows/"
    # The API expects fields like name, data, etc. We'll attempt to mirror that.
    payload = {
        "name": new_name or example_flow_json.get("name", "imported_flow"),
        "description": example_flow_json.get("description", "Imported example flow via POC"),
        "data": example_flow_json.get("data") or example_flow_json,  # some samples wrap JSON differently
        # project_id omitted -> default project
    }
    r = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return r.json()

def run_flow(flow_id, run_input):
    url = f"{LANGFLOW_URL}/api/v1/run/{flow_id}"
    r = requests.post(url, headers=HEADERS, json=run_input, timeout=120)
    r.raise_for_status()
    return r.json()

def main():
    print("1) Listing built-in example flows...")
    examples = list_basic_examples()
    if not examples:
        raise SystemExit("No basic examples found. Ensure your LangFlow version supports basic_examples endpoint.")
    # Pick first example (usually 'Basic Prompting' in many installs)
    example = examples[0]
    print("Found example:", example.get("name") or example.get("title") or "Unnamed")
    # To be safe, fetch the full example flow object if the listing returns only metadata
    # Some endpoints return the full flow; some only metadata. Try to get full JSON.
    # If example contains 'data' assume it's fully returned.
    example_flow_json = example
    # create/import the flow
    new_flow_resp = create_flow_from_example(example_flow_json, new_name=f"poc-{int(time.time())}")
    print("Created flow response (summary):", {k: new_flow_resp.get(k) for k in ("id","name")})
    flow_id = new_flow_resp.get("id")
    if not flow_id:
        # sometimes the create returns the full flow under 'data' or returns an object with nested data
        flow_id = new_flow_resp.get("data", {}).get("id") or new_flow_resp.get("flow", {}).get("id")
    if not flow_id:
        raise SystemExit("Could not determine created flow id. Inspect create response: " + json.dumps(new_flow_resp, indent=2))
    print("Running flow id:", flow_id)

    # Prepare input according to Basic Prompting example shape.
    # Many basic prompt flows expect an "input_value" parameter representing chat input.
    run_input = {
        "input_value": "Give me a concise summary of why LangFlow is useful for rapid prototyping AI workflows."
    }

    result = run_flow(flow_id, run_input)
    print("Run result (raw):")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
