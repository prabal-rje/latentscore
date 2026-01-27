import asyncio

from data_work.lib.baselines import get_baseline
from data_work.lib.config_batcher import (
    ConfigBatcher,
    build_batch_prompt,
    build_batch_response_model,
)


def _payload_for(vibe: str):
    baseline = get_baseline("random")
    payload = baseline.generate(vibe)
    return payload.model_copy(update={"thinking": f"just:{vibe}", "title": f"title:{vibe}"})


def test_baseline_generates_title() -> None:
    baseline = get_baseline("random")
    payload = baseline.generate("mellow dawn")
    assert payload.title


def test_build_batch_response_model_fields():
    response_model = build_batch_response_model(3)
    payloads = [_payload_for("a"), _payload_for("b"), _payload_for("c")]
    data = {
        "generated_config_0": payloads[0].model_dump(),
        "generated_config_1": payloads[1].model_dump(),
        "generated_config_2": payloads[2].model_dump(),
    }
    parsed = response_model.model_validate(data)
    assert parsed.generated_config_0.thinking == "just:a"
    assert parsed.generated_config_2.thinking == "just:c"
    assert parsed.generated_config_0.title == "title:a"


def test_config_batcher_batches_in_order():
    calls = []

    async def call_batch(vibes):
        calls.append(list(vibes))
        return [_payload_for(vibe) for vibe in vibes]

    async def run():
        # Use num_workers=1 for deterministic batching behavior in tests
        batcher = ConfigBatcher(batch_size=2, max_wait=0.01, call_batch=call_batch, num_workers=1)
        results = await asyncio.gather(
            batcher.submit("one"),
            batcher.submit("two"),
            batcher.submit("three"),
        )
        await batcher.aclose()
        return results

    results = asyncio.run(run())
    assert [r.thinking for r in results] == ["just:one", "just:two", "just:three"]
    assert calls[0] == ["one", "two"]
    assert calls[1] == ["three"]


def test_build_batch_prompt_places_vibes_last():
    prompt = build_batch_prompt(["alpha", "beta"], ["generated_config_0", "generated_config_1"])
    assert prompt.rstrip().endswith("</vibe_input>")
    assert "<vibe_input index=0>alpha</vibe_input>" in prompt
