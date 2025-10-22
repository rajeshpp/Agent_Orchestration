from src import prompt_templates

def test_triage_prompt_has_symptoms_placeholder():
    assert "{symptoms}" in prompt_templates.TRIAGE_PROMPT