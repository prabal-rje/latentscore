from latentscore.synth import _butter_cached


def test_butter_cache_key_stability() -> None:
    b1, a1 = _butter_cached("low", 0.5)
    b2, a2 = _butter_cached("low", 0.5)
    assert b1 is b2
    assert a1 is a2
