import json
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
