from __future__ import annotations

from latentscore import FirstAudioSpinner
from latentscore.spinner import Spinner


def test_spinner_disabled_is_noop() -> None:
    spinner = Spinner("Downloading", enabled=False)
    spinner.start()
    spinner.update("Still downloading")
    spinner.stop()


def test_first_audio_spinner_hooks_noop_when_disabled() -> None:
    spinner = FirstAudioSpinner(enabled=False)
    hooks = spinner.hooks()
    assert hooks.on_stream_start
    hooks.on_stream_start()
    assert hooks.on_first_audio_chunk
    hooks.on_first_audio_chunk()
    assert hooks.on_stream_end
    hooks.on_stream_end()
