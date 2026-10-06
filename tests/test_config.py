import pytest

from shared.config import SETTINGS, load_environment


def clear(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)


def test_dotenv_quotes_comments_bom_and_allowlist(tmp_path, monkeypatch):
    clear(monkeypatch)
    path = tmp_path / ".env"
    path.write_text('\ufeffGROQ_API_KEY="fixture-only" # comment\nexport GROQ_MODEL=fixture-model # comment\nPATH=bad\nHF_TOKEN=\n', encoding="utf-8")
    previous_path = __import__('os').environ.get("PATH")
    load_environment(path)
    import os
    assert os.environ["GROQ_API_KEY"] == "fixture-only"
    assert os.environ["GROQ_MODEL"] == "fixture-model"
    assert os.environ["PATH"] == previous_path
    assert os.environ["HF_TOKEN"] == ""


def test_existing_environment_has_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "existing-model")
    path = tmp_path / ".env"
    path.write_text("GROQ_MODEL=file-model\n", encoding="utf-8")
    load_environment(path)
    import os
    assert os.environ["GROQ_MODEL"] == "existing-model"


def test_invalid_quote_does_not_expose_secret(tmp_path, monkeypatch):
    clear(monkeypatch)
    path = tmp_path / ".env"
    path.write_text('GROQ_API_KEY="fixture-private\n', encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_environment(path)
    assert "fixture-private" not in str(error.value)
