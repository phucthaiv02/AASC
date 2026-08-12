from aas_nim_validation.config import DEFAULT_MODELS, Settings


def test_local_nim_does_not_require_api_key(monkeypatch):
    monkeypatch.setenv("NIM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.delenv("NIM_MODELS", raising=False)

    settings = Settings.from_env()

    assert settings.api_key == "not-used"
    assert settings.models == DEFAULT_MODELS


def test_models_are_parsed(monkeypatch):
    monkeypatch.setenv("NIM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("NIM_MODELS", " model/a , model/b ")

    assert Settings.from_env().models == ("model/a", "model/b")

