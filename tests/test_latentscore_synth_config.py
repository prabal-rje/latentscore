from latentscore.synth import MusicConfig


def test_synth_music_config_from_dict_flattens_layers() -> None:
    data = {
        "tempo": 0.4,
        "layers": {"bass": "drone", "pad": "warm_slow"},
        "unknown_key": "ignored",
    }
    config = MusicConfig.from_dict(data)
    assert config.bass == "drone"
    assert config.pad == "warm_slow"
