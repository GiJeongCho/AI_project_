import pytest
from fastapi.testclient import TestClient
from pytest_mock.plugin import MockerFixture
from src.api import app
from src.api_commons import SINGLE_MODE
from src.v1.schemas import GECResponse

client = TestClient(app)


def test_single_mode_enabled():
    assert SINGLE_MODE


def test_healthcheck_path():
    response = client.get("/docs")
    assert response.status_code == 200


def test_if_text_required():
    response = client.post("/gec", data={"text": ""})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "text,raise_error,status_code", [("1", True, 422), ("a", True, 200)]
)
def test_raise_error_option(
    mocker: MockerFixture, text: str, raise_error: bool, status_code: int
):
    mocker.patch("src.v1.chatgpt.get_gpt_result", return_value="")
    response = client.post(
        "/gec", data={"text": text, "raise_error": str(raise_error)}
    )
    assert response.status_code == status_code


@pytest.mark.parametrize(
    "text_input, text_output", [("It was midnigh", "It was midnight.")]
)
def test_response_model(mocker: MockerFixture, text_input: str, text_output: str):
    # patch output sentence
    mocker.patch("src.v1.chatgpt.get_gpt_result", return_value=text_output)
    response = client.post(
        "/gec", data={"text": text_input, "raise_error": str(True)}
    )
    assert response.status_code == 200

    # test if response matches schema
    response_model = GECResponse.model_validate(response.json())
    assert response_model.text == text_input
    assert response_model.correct_text == text_output
