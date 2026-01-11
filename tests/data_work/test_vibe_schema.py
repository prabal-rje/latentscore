from data_work.vibe_schema import (
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
        xs_vibe="short",
        sm_vibe="short",
        m_vibe="short",
        lg_vibe="short",
        xl_vibe="short",
    )
    obj = VibeObject(
        vibe_index=0,
        text_page="Page 0",
        characters=[CharacterVibes(character_name=character, character_perceived_vibes=[vibe])],
        scene_vibes=[vibe],
        tags=["too many words here now"],
    )
    response = VibeResponse(vibes=[obj])
    assert len(response.vibes[0].characters[0].character_name) <= MAX_SHORT_FIELD_CHARS
    assert response.vibes[0].tags[0].split() == ["too", "many", "words"]
