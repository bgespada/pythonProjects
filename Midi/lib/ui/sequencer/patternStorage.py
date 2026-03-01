import json
from pathlib import Path
from typing import Any


def save_pattern(file_path: str, pattern: dict[str, Any]) -> None:
    """Save a sequencer pattern dictionary as JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(pattern, f, indent=2)


def load_pattern(file_path: str) -> dict[str, Any]:
    """Load and return a sequencer pattern dictionary from JSON."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid pattern format")
    return data
