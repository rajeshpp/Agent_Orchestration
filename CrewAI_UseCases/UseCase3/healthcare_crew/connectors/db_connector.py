def fetch_patient_history(patient_id):
    # STUB: Simulate database call
    return {
        "patient_id": patient_id,
        "history": [
            {"condition": "Hypertension", "since": "2018"},
            {"condition": "Diabetes", "since": "2020"}
        ],
        "medications": ["Metformin", "Atenolol"],
        "allergies": ["Penicillin"]
    }
