import pytest
import pandas as pd
from unittest.mock import MagicMock
from mic_toolkit.synthetic.generation import Generator


@pytest.fixture
def generator():
    endpoint = "http://localhost:11434"
    model = "model"
    return Generator(endpoint, model)


def test_classify_text_correct_label(generator):
    labels = ["label1", "label2"]
    text = "sample text"
    generator.create_system_prompt = MagicMock(return_value="system prompt")
    generator.client.chat = MagicMock(return_value={"message": {"content": "label1"}})
    result = generator.generate_labels(labels, pd.DataFrame([text]), max_tries=5)
    assert result["label"].iloc[0] == "label1"


def test_classify_text_exceeds_max_tries(generator):
    labels = ["label1", "label2"]
    text = "sample text"
    generator.create_system_prompt = MagicMock(return_value="system prompt")
    generator.client.chat = MagicMock(
        side_effect=[{"message": {"content": "wrong label"}}] * 5
        + [{"message": {"content": "wrong label"}}]
    )
    result = generator.generate_labels(
        labels, pd.DataFrame({"text": [text]}), max_tries=5
    )
    assert result["label"].iloc[0] == "wrong label"


def test_generate_text(generator):
    text = "sample text"
    generator.client.chat = MagicMock(
        return_value={"message": {"content": "generated response"}}
    )
    result = generator.generate_text(pd.DataFrame({"text": [text]}))
    assert result["output"].iloc[0] == "generated response"
