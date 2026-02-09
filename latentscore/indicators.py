from __future__ import annotations

import sys
from typing import IO

from .main import ModelLoadRole, RenderHooks, Streamable, StreamContent, StreamEvent, StreamHooks
from .models import EXTERNAL_PREFIX, ExternalModelSpec, ModelSpec
from .spinner import Spinner, render_error


class RichIndicator:
    """Render/stream indicator that maps hooks to spinner updates."""

    def __init__(
        self,
        *,
        stream: IO[str] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._stream = stream or sys.stderr
        self._spinner = Spinner("Starting", stream=self._stream, enabled=enabled)
        self._started = False
        self._stopped = False

    def render_hooks(self) -> RenderHooks:
        return RenderHooks(
            on_start=self._on_render_start,
            on_model_start=self._on_render_model_start,
            on_model_end=self._on_render_model_end,
            on_synth_start=self._on_render_synth_start,
            on_synth_end=self._on_render_synth_end,
            on_end=self._on_render_end,
            on_error=self._on_render_error,
        )

    def stream_hooks(self) -> StreamHooks:
        return StreamHooks(
            on_event=self._on_stream_event,
            on_model_load_start=self._on_stream_model_load_start,
            on_model_load_end=self._on_stream_model_load_end,
            on_stream_start=self._on_stream_start,
            on_item_resolve_start=self._on_item_resolve_start,
            on_item_resolve_success=self._on_item_resolve_success,
            on_item_resolve_error=self._on_item_resolve_error,
            on_first_config_ready=self._on_first_config_ready,
            on_first_audio_chunk=self._on_first_audio_chunk,
            on_stream_end=self._on_stream_end,
            on_error=self._on_stream_error,
        )

    def _start(self, message: str) -> None:
        if self._stopped:
            return
        if not self._started:
            self._spinner.update(message)
            self._spinner.start()
            self._started = True
            return
        self._spinner.update(message)

    def _stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._spinner.stop()

    def _on_render_start(self) -> None:
        self._start("Preparing render")

    def _on_render_model_start(self, model: ModelSpec) -> None:
        if _is_external_model(model):
            self._start("Generating config (LLM)")
        else:
            self._start("Loading model")

    def _on_render_model_end(self, model: ModelSpec) -> None:
        _ = model
        self._start("Generating music")

    def _on_render_synth_start(self) -> None:
        self._start("Generating music")

    def _on_render_synth_end(self) -> None:
        self._start("Finalizing audio")

    def _on_render_end(self) -> None:
        self._stop()

    def _on_render_error(self, exc: Exception) -> None:
        self._stop()
        render_error("render", exc, stream=self._stream)

    def _on_stream_event(self, event: StreamEvent) -> None:
        _ = event

    def _on_stream_model_load_start(self, model: ModelSpec, role: ModelLoadRole) -> None:
        _ = role
        if _is_external_model(model):
            self._start("Preparing LLM session")
        else:
            self._start("Loading model")

    def _on_stream_model_load_end(self, model: ModelSpec, role: ModelLoadRole) -> None:
        _ = model
        _ = role
        if not self._started:
            self._start("Preparing stream")

    def _on_stream_start(self) -> None:
        self._start("Preparing stream")

    def _on_item_resolve_start(self, index: int, item: Streamable | StreamContent) -> None:
        label = _item_action(item)
        self._start(f"{label} (item {index + 1})")

    def _on_item_resolve_success(self, index: int, item: Streamable | StreamContent) -> None:
        _ = index
        _ = item

    def _on_item_resolve_error(
        self,
        index: int,
        item: Streamable | StreamContent,
        exc: Exception,
        fallback: object | None,
    ) -> None:
        _ = index
        _ = item
        _ = exc
        _ = fallback
        self._start("Config failed; applying fallback")

    def _on_first_config_ready(self, index: int, item: Streamable | StreamContent) -> None:
        _ = index
        _ = item
        self._start("Generating music")

    def _on_first_audio_chunk(self) -> None:
        self._start("Streaming audio")

    def _on_stream_end(self) -> None:
        self._stop()

    def _on_stream_error(self, exc: Exception) -> None:
        self._stop()
        render_error("stream", exc, stream=self._stream)


def _is_external_model(model: ModelSpec) -> bool:
    if isinstance(model, ExternalModelSpec):
        return True
    if isinstance(model, str):
        return model.startswith(EXTERNAL_PREFIX)
    return False


def _item_action(item: Streamable | StreamContent) -> str:
    content = item.content if isinstance(item, Streamable) else item
    if isinstance(content, str):
        return "Generating config (LLM)"
    return "Applying config"
