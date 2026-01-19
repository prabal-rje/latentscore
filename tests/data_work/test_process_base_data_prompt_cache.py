import importlib
import types

import pytest


@pytest.mark.asyncio
async def test_config_call_disables_cache_control(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("data_work.02b_generate_configs")
    captured: dict[str, object] = {}

    async def fake_completion(**kwargs: object):
        captured.update(kwargs)
        return types.SimpleNamespace(
            model_dump=lambda *args, **_kwargs: {
                "thinking": "",
                "title": "Quiet Dusk",
                "config": {},
                "palettes": [],
            }
        )

    monkeypatch.setattr(module, "litellm_structured_completion", fake_completion)

    await module.call_llm_for_config(
        vibe_text="quiet dusk",
        model="fake",
        api_key=None,
        api_base=None,
        system_prompt="SYSTEM",
        temperature=0.2,
        use_prompt_caching=False,
    )

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "<vibe>" in messages[1]["content"]
    assert captured["cache_control_injection_points"] is None


def test_parse_args_prompt_caching_flag(tmp_path) -> None:
    module = importlib.import_module("data_work.02b_generate_configs")
    args = module.build_parser().parse_args(
        [
            "--input-dir",
            str(tmp_path / "in"),
            "--output-dir",
            str(tmp_path / "out"),
            "--no-prompt-caching",
        ]
    )
    assert args.no_prompt_caching is True
