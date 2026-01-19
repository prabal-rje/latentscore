import pytest

from data_work.lib.llm_scorer import LLMScoreResult, score_config_with_llm_detailed_async


class DummyScorer:
    async def score_detailed_async(self, vibe: str, audio_path: str) -> LLMScoreResult:
        return LLMScoreResult(
            vibe_match=0.5,
            audio_quality=0.6,
            thinking="ok",
        )


@pytest.mark.asyncio
async def test_score_config_with_llm_detailed_async() -> None:
    result = await score_config_with_llm_detailed_async(
        vibe="test",
        config={"tempo": "medium"},
        scorer=DummyScorer(),
        duration=0.01,
    )
    assert result.vibe_match == 0.5
