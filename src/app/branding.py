from __future__ import annotations


def _slugify(name: str) -> str:
    cleaned: list[str] = []
    last_was_sep = True
    for ch in name.strip().lower():
        if ch.isalnum():
            cleaned.append(ch)
            last_was_sep = False
            continue
        if last_was_sep:
            continue
        cleaned.append("-")
        last_was_sep = True
    return "".join(cleaned).strip("-")


def _env_prefix(slug: str) -> str:
    cleaned: list[str] = []
    last_was_sep = True
    for ch in slug:
        if ch.isalnum():
            cleaned.append(ch.upper())
            last_was_sep = False
            continue
        if last_was_sep:
            continue
        cleaned.append("_")
        last_was_sep = True
    return "".join(cleaned).strip("_")


APP_NAME = "Sample App"
APP_SLUG = _slugify(APP_NAME) or "sample-app"
APP_ENV_PREFIX = _env_prefix(APP_SLUG) or "SAMPLE_APP"
