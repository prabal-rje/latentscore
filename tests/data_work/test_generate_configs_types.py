import importlib
from pathlib import Path

import pytest

from data_work.lib.llm_cache import SqliteCache
from data_work.lib.pipeline_models import ConfigGenerationRow, JsonDict, VibeRow


@pytest.mark.asyncio
async def test_generate_configs_returns_pydantic_row(
    tmp_path: Path,
) -> None:
    gen = importlib.import_module("data_work.02b_generate_configs")

    async def fake_config_call(_vibe_text: str) -> JsonDict:
        return {"thinking": "", "title": "Mist Drift", "config": {}, "palettes": []}

    row = VibeRow(
        dataset="demo",
        id_in_dataset="1",
        split="SFT-Train",
        vibe_index=0,
        text_page=0,
        vibe_scope="scene",
        character_name=None,
        vibe_level="sm",
        vibe_original="mist",
        vibe_noisy="mist",
        tags_original=[],
        tags_noisy=[],
        vibe_model="model",
    )
    cache = SqliteCache(tmp_path / "cache.sqlite")
    await cache.initialize()

    result = await gen.generate_configs_for_row(
        row,
        model="fake",
        api_base=None,
        prompt_hash="hash",
        num_candidates=1,
        temperature=0.0,
        seed=0,
        cache=cache,
        config_call=fake_config_call,
        max_retries=0,
        retry_base_delay=0.0,
    )

    assert isinstance(result, ConfigGenerationRow)
