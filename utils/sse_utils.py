import json
from typing import Any


def _sse_line(obj: dict[str, Any]) -> bytes:
    return (
        "data: " + json.dumps(obj, ensure_ascii=False, default=str) + "\n\n"
    ).encode("utf-8")
