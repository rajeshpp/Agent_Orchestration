def check_drug_interactions(drug_list):
    # STUB: Simulate a drug interaction API
    if "Atenolol" in drug_list and "Verapamil" in drug_list:
        return [{"pair": ("Atenolol", "Verapamil"), "risk": "High", "message": "Serious bradycardia risk."}]
    return []
