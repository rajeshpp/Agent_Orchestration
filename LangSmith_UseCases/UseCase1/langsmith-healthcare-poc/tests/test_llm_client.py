import pytest
from unittest.mock import patch
from src.llm_client import triage_symptoms


@patch("src.llm_client.openai.ChatCompletion.create")
def test_triage_parses_json(mock_create):
    mock_create.return_value = {
        "choices": [
            {"message": {"content": '{"triage_level":"SELF_CARE","rationale":"mild symptoms","recommended_next_step":"Stay hydrated"}'}}
        ]
    }

    out = triage_symptoms("fever")
    assert out["triage_level"] == "SELF_CARE"
    assert "recommended_next_step" in out