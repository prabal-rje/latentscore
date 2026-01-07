from latentscore.models import ExternalModelSpec, resolve_model
from latentscore.providers.litellm import LiteLLMAdapter


def test_resolve_model_caches_builtin_instances() -> None:
    first = resolve_model("fast")
    second = resolve_model("fast")
    assert first is second


def test_resolve_model_external_prefix() -> None:
    model = resolve_model("external:gemini/gemini-3-flash-preview")
    assert isinstance(model, LiteLLMAdapter)


def test_resolve_model_external_spec() -> None:
    spec = ExternalModelSpec(model="gemini/gemini-3-flash-preview", api_key=None)
    model = resolve_model(spec)
    assert isinstance(model, LiteLLMAdapter)
