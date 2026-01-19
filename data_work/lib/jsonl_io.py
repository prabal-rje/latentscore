"""JSONL helpers for data_work scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import JsonValue

JsonDict = dict[str, JsonValue]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            match row:
                case dict():
                    yield row
                case _:
                    pass
