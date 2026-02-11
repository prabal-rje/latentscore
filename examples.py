"""Quick demo of the latentscore live API: async generator-driven streaming playback."""

import asyncio
from collections.abc import AsyncIterator

import latentscore as ls
from latentscore.config import Step


async def my_set() -> AsyncIterator[str | ls.MusicConfigUpdate]:
    """Yield vibes and config updates — the live engine crossfades between them."""

    # Start with a vibe
    yield "warm jazz cafe at midnight"
    await asyncio.sleep(8)

    # Absolute config update: switch to bright electronic
    yield ls.MusicConfigUpdate(tempo="fast", brightness="very_bright", rhythm="electronic")
    await asyncio.sleep(8)

    # Relative step update: dial brightness back down, add more echo
    yield ls.MusicConfigUpdate(brightness=Step(-2), echo=Step(+1))


# Wire the generator to live playback — streams audio to speakers as it goes
session = ls.live(my_set(), transition_seconds=2.0)
session.play(seconds=30)
