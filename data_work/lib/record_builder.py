"""Flatten vibe responses into per-level rows."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from data_work.lib.vibe_schema import VibeDescriptor, VibeResponse

VibeLevel = Literal["xl", "lg", "m", "sm", "xs"]
VibeScope = Literal["character", "scene"]


class VibeRow(TypedDict):
    dataset: str
    id_in_dataset: Any
    split: str
    vibe_index: int
    text_page: tuple[int, int]
    vibe_scope: VibeScope
    character_name: str | None
    vibe_level: VibeLevel
    vibe_original: str
    vibe_noisy: str
    tags_original: list[str]
    tags_noisy: list[str]


_LEVELS: tuple[tuple[VibeLevel, str], ...] = (
    ("xl", "xl_vibe"),
    ("lg", "lg_vibe"),
    ("m", "m_vibe"),
    ("sm", "sm_vibe"),
    ("xs", "xs_vibe"),
)


def _iter_levels(
    original: VibeDescriptor, noisy: VibeDescriptor
) -> list[tuple[VibeLevel, str, str]]:
    rows: list[tuple[VibeLevel, str, str]] = []
    for level, attr in _LEVELS:
        original_value = getattr(original, attr)
        noisy_value = getattr(noisy, attr)
        assert isinstance(original_value, str)
        assert isinstance(noisy_value, str)
        rows.append((level, original_value, noisy_value))
    return rows


def _pair_descriptors(
    original: list[VibeDescriptor],
    noisy: list[VibeDescriptor],
) -> list[tuple[VibeDescriptor, VibeDescriptor]]:
    pairs: list[tuple[VibeDescriptor, VibeDescriptor]] = []
    max_len = max(len(original), len(noisy))
    for idx in range(max_len):
        original_item = original[idx] if idx < len(original) else noisy[idx]
        noisy_item = noisy[idx] if idx < len(noisy) else original_item
        pairs.append((original_item, noisy_item))
    return pairs


def build_vibe_rows(
    *,
    dataset: str,
    id_in_dataset: Any,
    split_name: str,
    original: VibeResponse,
    noisy: VibeResponse,
) -> list[VibeRow]:
    rows: list[VibeRow] = []
    for original_obj, noisy_obj in zip(original.vibes, noisy.vibes):
        base = {
            "dataset": dataset,
            "id_in_dataset": id_in_dataset,
            "split": split_name,
            "vibe_index": original_obj.vibe_index,
            "text_page": original_obj.text_page,
            "tags_original": list(original_obj.tags),
            "tags_noisy": list(noisy_obj.tags),
        }
        for original_char, noisy_char in zip(original_obj.characters, noisy_obj.characters):
            for orig_desc, noisy_desc in _pair_descriptors(
                original_char.character_perceived_vibes,
                noisy_char.character_perceived_vibes,
            ):
                for level, original_text, noisy_text in _iter_levels(orig_desc, noisy_desc):
                    rows.append(
                        {
                            **base,
                            "vibe_scope": "character",
                            "character_name": original_char.character_name,
                            "vibe_level": level,
                            "vibe_original": original_text,
                            "vibe_noisy": noisy_text,
                        }
                    )
        for orig_desc, noisy_desc in _pair_descriptors(
            original_obj.scene_vibes, noisy_obj.scene_vibes
        ):
            for level, original_text, noisy_text in _iter_levels(orig_desc, noisy_desc):
                rows.append(
                    {
                        **base,
                        "vibe_scope": "scene",
                        "character_name": None,
                        "vibe_level": level,
                        "vibe_original": original_text,
                        "vibe_noisy": noisy_text,
                    }
                )
    return rows
