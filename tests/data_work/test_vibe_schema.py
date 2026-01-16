from data_work.lib.vibe_schema import (
    MAX_SHORT_FIELD_CHARS,
    CharacterVibes,
    VibeDescriptor,
    VibeObject,
    VibeResponse,
    paginate_text,
)


def test_paginate_text_inserts_page_markers() -> None:
    text = "word " * 2100
    paged = paginate_text(text, max_tokens=2000, page_tokens=1000)
    assert "(Page 0)" in paged
    assert "(Page 1)" in paged
    assert "word" in paged


def test_schema_trims_short_fields() -> None:
    character = "x" * (MAX_SHORT_FIELD_CHARS + 10)
    vibe = VibeDescriptor(
        xl_vibe="short",
        lg_vibe="short",
        m_vibe="short",
        sm_vibe="short",
        xs_vibe="short",
    )
    obj = VibeObject(
        vibe_index=0,
        text_page=0,
        characters=[CharacterVibes(character_name=character, character_perceived_vibes=[vibe])],
        scene_vibes=[vibe],
        tags=["too many words here now"],
    )
    response = VibeResponse(vibes=[obj])
    assert len(response.vibes[0].characters[0].character_name) <= MAX_SHORT_FIELD_CHARS
    assert response.vibes[0].tags[0].split() == ["too", "many", "words"]


def test_character_name_defaults_to_none_when_empty() -> None:
    vibe = VibeDescriptor(
        xl_vibe="short",
        lg_vibe="short",
        m_vibe="short",
        sm_vibe="short",
        xs_vibe="short",
    )
    character = CharacterVibes(character_name=" ", character_perceived_vibes=[vibe])
    assert character.character_name == "none"


def test_vibe_object_single_page() -> None:
    """Ensure text_page accepts single integer."""
    obj = VibeObject(
        vibe_index=0,
        text_page=5,  # Single page, not a range
        characters=[],
        scene_vibes=[],
        tags=["test"],
    )
    assert obj.text_page == 5


def test_vibe_object_rejects_range_takes_first() -> None:
    """Range inputs should take first page (fallback behavior)."""
    obj = VibeObject(
        vibe_index=0,
        text_page=[3, 7],  # type: ignore[arg-type]  # Range given, should take first
        characters=[],
        scene_vibes=[],
        tags=["test"],
    )
    assert obj.text_page == 3  # Takes first page


def test_text_page_parses_string_takes_first() -> None:
    """String page references should take first number only."""
    vibe = VibeDescriptor(
        xl_vibe="short",
        lg_vibe="short",
        m_vibe="short",
        sm_vibe="short",
        xs_vibe="short",
    )
    obj = VibeObject(
        vibe_index=0,
        text_page="Page 3-4",  # type: ignore[arg-type]
        characters=[],
        scene_vibes=[vibe],
        tags=["tag"],
    )
    assert obj.text_page == 3  # Takes first page only
