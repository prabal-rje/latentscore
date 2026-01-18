"""Shared schema + helpers for vibe extraction."""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Iterable, Sequence, TypeVar

from common.prompts import build_vibe_extraction_prompt
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_LONG_FIELD_CHARS = 1_000
MAX_SHORT_FIELD_CHARS = 100
DEFAULT_MAX_INPUT_TOKENS = 100_000
DEFAULT_PAGE_TOKENS = 1_000

SPLITS: tuple[tuple[str, float], ...] = (
    ("SFT-Train", 0.60),
    ("SFT-Val", 0.10),
    ("GRPO", 0.15),
    ("TEST", 0.15),
)

T = TypeVar("T")


def _trim_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[:limit]
    return text


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def paginate_text(text: str, max_tokens: int, page_tokens: int) -> str:
    """Insert page markers every page_tokens using whitespace tokenization."""
    tokens = re.findall(r"\S+", text)
    tokens = tokens[:max_tokens]
    pages = [tokens[idx : idx + page_tokens] for idx in range(0, len(tokens), page_tokens)]
    chunks: list[str] = []
    for page_index, page_items in enumerate(pages):
        chunks.append(f"(Page {page_index})")
        chunks.append(" ".join(page_items))
    return "\n".join(chunks).strip()


def split_records(records: Sequence[T], seed: int) -> dict[str, list[T]]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    counts: list[tuple[str, int]] = []
    remaining = len(records)
    for name, ratio in SPLITS:
        count = int(len(records) * ratio)
        count = min(count, remaining)
        counts.append((name, count))
        remaining -= count
    if remaining:
        name, count = counts[0]
        counts[0] = (name, count + remaining)
    split_map: dict[str, list[T]] = {name: [] for name, _ in SPLITS}
    cursor = 0
    for name, count in counts:
        for idx in indices[cursor : cursor + count]:
            split_map[name].append(records[idx])
        cursor += count
    return split_map


class VibeDescriptor(BaseModel):
    """Five-step descriptive ladder from micro-label to rich detail."""

    model_config = ConfigDict(extra="forbid")

    xl_vibe: str = Field(
        ...,
        max_length=MAX_LONG_FIELD_CHARS,
        description=("Extra-large description (2-3 short lines) that feels vivid and cinematic."),
    )
    lg_vibe: str = Field(
        ...,
        max_length=MAX_LONG_FIELD_CHARS,
        description=(
            "Large description (2-3 sentences) adding nuance, tone shifts, and sensory cues."
        ),
    )
    m_vibe: str = Field(
        ...,
        max_length=MAX_LONG_FIELD_CHARS,
        description=(
            "Medium description (1-2 sentences) capturing the mood with one concrete detail."
        ),
    )
    sm_vibe: str = Field(
        ...,
        max_length=MAX_LONG_FIELD_CHARS,
        description=(
            "Short phrase (5-10 words) that slightly expands the vibe while staying compact."
        ),
    )
    xs_vibe: str = Field(
        ...,
        max_length=MAX_LONG_FIELD_CHARS,
        description=(
            "Ultra-short label (1-3 words) capturing the tiniest, most atomic vibe signal."
        ),
    )

    @field_validator("*", mode="before")
    @classmethod
    def _trim_long(cls, value: Any) -> str:
        return _trim_text(value, MAX_LONG_FIELD_CHARS)


class CharacterVibes(BaseModel):
    """Vibes as experienced or projected by a specific character."""

    model_config = ConfigDict(extra="forbid")

    character_name: str = Field(
        ...,
        max_length=MAX_SHORT_FIELD_CHARS,
        description=(
            "Character label from the text. Use names when known, otherwise "
            "labels like 'anonymous', 'stranger 1', or '3rd person'."
        ),
    )
    character_perceived_vibes: list[VibeDescriptor] = Field(
        ...,
        description=(
            "List of vibe descriptors reflecting how this character feels or is perceived "
            "in the scene."
        ),
    )

    @field_validator("character_name", mode="before")
    @classmethod
    def _trim_name(cls, value: Any) -> str:
        text = _trim_text(value, MAX_SHORT_FIELD_CHARS)
        return text or "none"


class VibeObject(BaseModel):
    """Single vibe object tied to a page reference with character + scene vibes."""

    model_config = ConfigDict(extra="forbid")

    vibe_index: int = Field(
        ...,
        ge=0,
        description=("0-based index that increases by 1 in the order the vibe objects appear."),
    )
    text_page: int = Field(
        ...,
        ge=0,
        description=(
            "Single 0-based page number matching ONE (Page N) marker. "
            "IMPORTANT: You MUST process ONE page at a time. Page ranges are NOT allowed. "
            "If content spans multiple pages, create separate vibe objects for each page."
        ),
    )
    characters: list[CharacterVibes] = Field(
        ...,
        description=(
            "All characters mentioned in this vibe segment, each with their perceived vibes."
        ),
    )
    scene_vibes: list[VibeDescriptor] = Field(
        ...,
        description=("Overall scene vibes that apply beyond any single character."),
    )
    tags: list[str] = Field(
        ...,
        description=(
            "Atomic tags (topics/places/feelings) using 1-3 words each, lowercase when possible."
        ),
    )

    @field_validator("text_page", mode="before")
    @classmethod
    def _parse_page(cls, value: Any) -> int:
        """Parse page number, rejecting ranges."""
        match value:
            case None:
                return 0
            case list() | tuple():
                items: list[Any] = list(value)
                if not items:
                    return 0
                # If given a range, take the first page only
                # The prompt explicitly forbids ranges, so this is a fallback
                return int(items[0])
            case str():
                numbers = [int(item) for item in re.findall(r"\d+", value)]
                return numbers[0] if numbers else 0
            case _:
                return int(value)

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> list[str]:
        """Normalize tags to a list of trimmed strings."""
        match value:
            case None:
                return []
            case str():
                items = [value]
            case _:
                items = list(value)
        normalized: list[str] = []
        for item in items:
            text = _trim_text(item, MAX_SHORT_FIELD_CHARS)
            text = _limit_words(text, 3)
            if text:
                normalized.append(text)
        return normalized


class VibeResponse(BaseModel):
    """Top-level response containing all vibe objects for a text."""

    model_config = ConfigDict(extra="forbid")

    vibes: list[VibeObject] = Field(
        ...,
        description=("Ordered list of vibe objects. Keep vibe_index sequential starting at 0."),
    )


def normalize_vibe_indices(vibes: Iterable[VibeObject]) -> list[VibeObject]:
    return [vibe.model_copy(update={"vibe_index": index}) for index, vibe in enumerate(vibes)]


def schema_signature() -> str:
    schema = VibeResponse.model_json_schema()
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def schema_hash() -> str:
    signature = schema_signature().encode("utf-8")
    return hashlib.sha256(signature).hexdigest()


def build_vibe_prompt() -> str:
    schema = schema_signature()
    return build_vibe_extraction_prompt(schema)
