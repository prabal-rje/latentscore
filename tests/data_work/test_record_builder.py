from data_work.lib.record_builder import build_vibe_rows
from data_work.lib.vibe_schema import (
    CharacterVibes,
    VibeDescriptor,
    VibeObject,
    VibeResponse,
)


def test_build_vibe_rows_expands_levels() -> None:
    original_desc = VibeDescriptor(
        xl_vibe="xl",
        lg_vibe="lg",
        m_vibe="m",
        sm_vibe="sm",
        xs_vibe="xs",
    )
    noisy_desc = VibeDescriptor(
        xl_vibe="xl_n",
        lg_vibe="lg_n",
        m_vibe="m_n",
        sm_vibe="sm_n",
        xs_vibe="xs_n",
    )
    original_obj = VibeObject(
        vibe_index=0,
        text_page=(0, 0),
        characters=[
            CharacterVibes(
                character_name="anon",
                character_perceived_vibes=[original_desc],
            )
        ],
        scene_vibes=[original_desc],
        tags=["tag"],
    )
    noisy_obj = VibeObject(
        vibe_index=0,
        text_page=(0, 0),
        characters=[
            CharacterVibes(
                character_name="anon",
                character_perceived_vibes=[noisy_desc],
            )
        ],
        scene_vibes=[noisy_desc],
        tags=["tag_noisy"],
    )
    rows = build_vibe_rows(
        dataset="ds",
        id_in_dataset="1",
        split_name="SFT-Train",
        original=VibeResponse(vibes=[original_obj]),
        noisy=VibeResponse(vibes=[noisy_obj]),
    )
    assert len(rows) == 10
    scene_xl = next(
        row for row in rows if row["vibe_scope"] == "scene" and row["vibe_level"] == "xl"
    )
    assert scene_xl["vibe_original"] == "xl"
    assert scene_xl["vibe_noisy"] == "xl_n"
    assert scene_xl["tags_original"] == ["tag"]
    assert scene_xl["tags_noisy"] == ["tag_noisy"]
