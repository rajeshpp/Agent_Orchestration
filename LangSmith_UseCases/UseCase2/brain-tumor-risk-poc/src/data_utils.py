import pandas as pd

def normalize_patient_data(raw_data: dict) -> dict:
    """
    Perform light data cleaning & normalization.
    Convert keys to lowercase, strip whitespace, remove nulls.
    """
    cleaned = {}
    for k, v in raw_data.items():
        if v and str(v).strip():
            cleaned[k.lower().strip()] = str(v).strip()
    return cleaned


def summarize_patient_data(data: dict) -> str:
    """
    Convert patient dictionary into natural-language summary text.
    """
    df = pd.DataFrame([data])
    summary = "; ".join(f"{k}: {v}" for k, v in df.iloc[0].items())
    return f"Patient summary: {summary}"
