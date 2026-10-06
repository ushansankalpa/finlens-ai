"""Load known settings from the workspace .env without logging secret values."""
from __future__ import annotations

import os
import re
from pathlib import Path

SETTINGS = {"GROQ_API_KEY", "GROQ_MODEL", "TEACHER_MODEL", "HF_TOKEN"}


def load_environment(path: str | Path | None = None) -> None:
    path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in SETTINGS:
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end+1:].strip() and not value[end+1:].strip().startswith("#")):
                raise ValueError(f"Invalid quoted .env setting: {name}")
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        # No interpolation, shell evaluation, or override of existing environment values.
        os.environ.setdefault(name, value)
