import numpy as np

from latentscore.dx import Audio, Playlist, Track


def test_audio_normalizes_samples() -> None:
    audio = Audio(samples=np.array([2.0, -2.0], dtype=np.float32))
    assert float(audio.samples.max()) <= 1.0


def test_track_to_streamable() -> None:
    track = Track(content="warm sunrise", duration=1.5, transition=0.2)
    streamable = track.to_streamable()
    assert streamable.duration == 1.5
    assert streamable.transition_duration == 0.2


def test_playlist_roundtrip() -> None:
    playlist = Playlist(tracks=(Track(content="warm sunrise"),))
    assert len(playlist.tracks) == 1
