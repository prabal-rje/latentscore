from latentscore import models as model_mod
from latentscore.models import ExternalModelSpec, resolve_model
from latentscore.providers.litellm import LiteLLMAdapter


def test_resolve_model_caches_builtin_instances() -> None:
    first = resolve_model("fast")
    second = resolve_model("fast")
    assert first is second


def _unwrap_external(model: object) -> object:
    if isinstance(model, model_mod._FallbackModel):
        return model._primary
    return model


def test_resolve_model_external_prefix() -> None:
    model = resolve_model("external:gemini/gemini-3-flash-preview")
    assert isinstance(_unwrap_external(model), LiteLLMAdapter)


def test_resolve_model_external_spec() -> None:
    spec = ExternalModelSpec(model="gemini/gemini-3-flash-preview", api_key=None)
    model = resolve_model(spec)
    assert isinstance(_unwrap_external(model), LiteLLMAdapter)
