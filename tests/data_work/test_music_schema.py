from data_work.lib.music_schema import MusicConfigPromptPayload, schema_hash


def test_music_schema_validates() -> None:
    payload = {
        "justification": "Calm, slow, and spacious to match a quiet midnight mood.",
        "config": {
            "tempo": "slow",
            "root": "d",
            "mode": "minor",
            "brightness": "dark",
            "space": "large",
            "density": 3,
            "bass": "drone",
            "pad": "ambient_drift",
            "melody": "minimal",
            "rhythm": "none",
            "texture": "shimmer",
            "accent": "none",
            "motion": "slow",
            "attack": "soft",
            "stereo": "wide",
            "depth": True,
            "echo": "subtle",
            "human": "natural",
            "grain": "warm",
        },
    }
    parsed = MusicConfigPromptPayload.model_validate(payload)
    assert parsed.config.tempo == "slow"
    assert parsed.config.density == 3


def test_music_schema_hash_stable() -> None:
    assert schema_hash() == schema_hash()
