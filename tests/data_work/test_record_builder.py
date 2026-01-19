from data_work.lib.record_builder import build_vibe_rows
from data_work.lib.vibe_schema import VibeDescriptor, VibeObject, VibeResponse


def test_build_vibe_rows_returns_typed_rows() -> None:
    descriptor = VibeDescriptor(
        xl_vibe="xl",
        lg_vibe="lg",
        m_vibe="m",
        sm_vibe="sm",
        xs_vibe="xs",
    )
    response = VibeResponse(
        vibes=[
            VibeObject(
                vibe_index=0,
                text_page=1,
                characters=[],
                scene_vibes=[descriptor],
                tags=["tag"],
            )
        ]
    )
    rows = build_vibe_rows(
        dataset="demo",
        id_in_dataset="1",
        split_name="SFT-Train",
        original=response,
        noisy=response,
    )
    assert rows[0].text_page == 1
    assert rows[0].vibe_level == "xl"
