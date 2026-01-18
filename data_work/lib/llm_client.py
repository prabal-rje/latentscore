"""Shared LLM helpers for data_work workflows."""

from __future__ import annotations

import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel

LOGGER = logging.getLogger("data_work.llm_client")

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

GEMMA_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}{% elif message['role'] == 'system' %}{{ '<start_of_turn>system\\n' + message['content'] + '<end_of_turn>\\n' }}{% elif message['role'] == 'assistant' %}{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"""

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


def is_qwen3_model(model_name: str) -> bool:
    return "qwen3" in model_name.lower()


def is_gemma_model(model_name: str) -> bool:
    return "gemma" in model_name.lower()


def normalize_tokenizer_for_model(tokenizer: Any, model_name: str) -> None:
    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token_id", None) is not None
    ):
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    if is_gemma_model(model_name):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE


def wrap_vibe_for_chat(vibe: str) -> str:
    return f"<vibe>{vibe}</vibe>"


def render_chat_prompt(
    *,
    system_prompt: str,
    user_prompt: str,
    tokenizer: Any,
    model_name: str,
    add_generation_prompt: bool,
    assistant: str | None = None,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if is_qwen3_model(model_name):
        kwargs["enable_thinking"] = False
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, **kwargs)

    if not any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    ):
        allowed_kwargs = {
            name
            for name, param in sig.parameters.items()
            if param.kind
            in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        }
        kwargs = {key: value for key, value in kwargs.items() if key in allowed_kwargs}

    if "conversation" in sig.parameters:
        param = sig.parameters["conversation"]
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            return tokenizer.apply_chat_template(messages, **kwargs)
        kwargs["conversation"] = messages
        return tokenizer.apply_chat_template(**kwargs)
    if "messages" in sig.parameters:
        param = sig.parameters["messages"]
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            return tokenizer.apply_chat_template(messages, **kwargs)
        kwargs["messages"] = messages
        return tokenizer.apply_chat_template(**kwargs)
    return tokenizer.apply_chat_template(messages, **kwargs)


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
        model_list = ", ".join(model for model, _ in models if model in required)
        raise SystemExit(
            f"API key required for model(s): {model_list}\n\n"
            f"Set the environment variable and try again:\n"
            f"  export {api_key_env}='your-key-here'\n\n"
            f"Or pass --api-key directly, or create a .env file."
        )
    return None


def check_env_keys_at_startup(
    *,
    required_keys: Sequence[tuple[str, str]],
    optional_keys: Sequence[tuple[str, str]] | None = None,
) -> dict[str, str | None]:
    """Check for required environment keys at startup and print helpful messages.

    Args:
        required_keys: List of (env_var_name, description) tuples that must be set.
        optional_keys: List of (env_var_name, description) tuples that are optional.

    Returns:
        Dict mapping env var names to their values (None for missing optional keys).

    Raises:
        SystemExit: If any required key is missing, with a helpful error message.
    """
    result: dict[str, str | None] = {}
    missing: list[tuple[str, str]] = []

    for key, desc in required_keys:
        value = os.environ.get(key)
        if value:
            result[key] = value
            LOGGER.debug("Found %s: %s", key, mask_api_key(value))
        else:
            missing.append((key, desc))

    if missing:
        lines = [
            "Missing required environment variable(s):",
            "",
        ]
        for key, desc in missing:
            lines.append(f"  {key}")
            lines.append(f"    {desc}")
            lines.append("")
        lines.append("Set the variable(s) and try again:")
        for key, _ in missing:
            lines.append(f"  export {key}='your-key-here'")
        lines.append("")
        lines.append("Or create a .env file with the variable(s).")
        raise SystemExit("\n".join(lines))

    if optional_keys:
        for key, desc in optional_keys:
            value = os.environ.get(key)
            result[key] = value
            if value:
                LOGGER.debug("Found optional %s: %s", key, mask_api_key(value))

    return result


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
    """Coerce various response formats into the expected model type."""
    # Check if already the target type first (can't be done in match easily)
    if isinstance(content, response_model):
        return content

    match content:
        case BaseModel():
            return response_model.model_validate(content.model_dump())
        case dict():
            return response_model.model_validate(content)
        case str():
            parsed = _parse_json_payload(content)
            return response_model.model_validate(parsed)
        case _:
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
        self._model_name = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        normalize_tokenizer_for_model(self._tokenizer, model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if self._device is not None:
            model = model.to(self._device)
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def format_chat_prompt(self, *, system_prompt: str, user_prompt: str) -> str:
        return render_chat_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tokenizer=self._tokenizer,
            model_name=self._model_name,
            add_generation_prompt=True,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device is not None:
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        effective_max_new_tokens = max_new_tokens or self._max_new_tokens
        effective_temperature = self._temperature if temperature is None else temperature
        do_sample = effective_temperature > 0
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=effective_max_new_tokens,
            do_sample=do_sample,
            temperature=effective_temperature if do_sample else 1.0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][prompt_len:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        text = self.generate_text(prompt)
        parsed = _parse_json_payload(text)
        return response_model.model_validate(parsed)
