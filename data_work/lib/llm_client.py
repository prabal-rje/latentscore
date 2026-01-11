"""Shared LLM helpers for data_work workflows."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel

LOGGER = logging.getLogger("data_work.llm_client")

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

T = TypeVar("T", bound=BaseModel)


class LLMResponseError(RuntimeError):
    """Raised when an LLM response cannot be parsed."""


def load_env_file(env_path: Path | None) -> None:
    candidates = [Path(".env")]
    resolved = env_path
    if resolved is None:
        for candidate in candidates:
            if candidate.is_file():
                resolved = candidate
                break
    if resolved is None or not resolved.is_file():
        return
    for line in resolved.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def normalize_model_and_base(model: str, api_base: str | None) -> tuple[str, str | None]:
    if model.startswith("http"):
        parsed = urlparse(model)
        if "openrouter.ai" in parsed.netloc and parsed.path:
            model = f"openrouter/{parsed.path.lstrip('/')}"
            api_base = api_base or DEFAULT_OPENROUTER_API_BASE
    if model.startswith("openrouter/") and api_base is None:
        api_base = DEFAULT_OPENROUTER_API_BASE
    return model, api_base


def requires_api_key(model: str, api_base: str | None) -> bool:
    match model:
        case _ if model.startswith("openrouter/"):
            return True
        case _ if model.startswith("openai/"):
            return True
        case _ if model.startswith("anthropic/"):
            return True
        case _:
            return api_base is not None and "openrouter.ai" in api_base


def mask_api_key(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) <= 8:
        return f"{cleaned[:1]}...{cleaned[-1:]}"
    return f"{cleaned[:4]}...{cleaned[-4:]}"


def resolve_api_key_for_models(
    *,
    api_key: str | None,
    api_key_env: str,
    models: Sequence[tuple[str, str | None]],
) -> str | None:
    if api_key:
        return api_key
    required = [model for model, api_base in models if requires_api_key(model, api_base)]
    env_value = os.environ.get(api_key_env)
    if env_value:
        if required:
            masked = mask_api_key(env_value)
            LOGGER.warning(
                "Using API key from %s: %s (pass --api-key to override).",
                api_key_env,
                masked,
            )
            return env_value
        return None
    if required:
        raise SystemExit(
            "API key required for model(s): "
            + ", ".join(model for model, _ in models if model in required)
        )
    return None


def format_prompt_json(system_prompt: str, user_prompt: str) -> str:
    return json.dumps(
        {
            "system": system_prompt,
            "user": user_prompt,
            "assistant": "",
        },
        ensure_ascii=False,
    )


def _parse_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidate = stripped
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise LLMResponseError("No JSON object found in LLM response.")
        candidate = stripped[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("LLM returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError("LLM JSON response was not an object.")
    return parsed


def _extract_litellm_content(response: Any) -> Any:
    match response:
        case {"choices": choices} if choices:
            message = choices[0].get("message", {})
            return message.get("content")
        case _ if hasattr(response, "choices"):
            choice = response.choices[0]
            message = choice.message
            return getattr(message, "parsed", None) or getattr(message, "content", None)
        case _:
            return None


def _coerce_response(
    *,
    content: Any,
    response_model: type[T],
    context: str,
) -> T:
    if isinstance(content, response_model):
        return content
    if isinstance(content, BaseModel):
        return response_model.model_validate(content.model_dump())
    if isinstance(content, dict):
        return response_model.model_validate(content)
    if isinstance(content, str):
        parsed = _parse_json_payload(content)
        return response_model.model_validate(parsed)
    raise LLMResponseError(f"{context} response missing structured content.")


async def litellm_structured_completion(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    response_model: type[T],
    api_key: str | None,
    api_base: str | None,
    model_kwargs: Mapping[str, Any],
    temperature: float = 0.0,
) -> T:
    try:
        import litellm  # type: ignore[import]
        from litellm import acompletion  # type: ignore[import]
    except ImportError as exc:
        LOGGER.warning("LiteLLM not installed: %s", exc)
        raise SystemExit("litellm is required. Install via data_work/requirements.txt.") from exc

    litellm.turn_off_message_logging = True
    litellm.disable_streaming_logging = True
    litellm.logging = False

    request: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "response_format": response_model,
    }
    request.update(model_kwargs)
    if api_key:
        request["api_key"] = api_key
    if api_base:
        request["api_base"] = api_base

    response = await acompletion(**request)
    content = _extract_litellm_content(response)
    return _coerce_response(
        content=content,
        response_model=response_model,
        context="LiteLLM",
    )


class LocalHFClient:
    """Simple local Hugging Face model wrapper for JSON outputs."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | None,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            LOGGER.warning("transformers/torch not installed: %s", exc)
            raise SystemExit(
                "Local models require transformers + torch. "
                "Install them in the data_work environment."
            ) from exc

        self._device = torch.device(device) if device else None
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if self._device is not None:
            model = model.to(self._device)
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def generate_text(self, prompt: str) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device is not None:
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        do_sample = self._temperature > 0
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=do_sample,
            temperature=self._temperature if do_sample else 1.0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][prompt_len:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        text = self.generate_text(prompt)
        parsed = _parse_json_payload(text)
        return response_model.model_validate(parsed)
