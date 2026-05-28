# AI Disclosure

This file documents AI use in this repository.

## Scope of AI tool usage

**Coding.** The code in this repository was written by a human and LLM coding assistants under human direction and oversight. Schema design (`MusicConfig` and its categorical labels) is the most prominent example; the same applies to scaffolding, refactoring, debugging, tests, packaging, the `doctor` CLI, and documentation. The human author wrote code directly, specified requirements, reviewed and edited AI-generated output, and made the final design decisions.

**Library components.** Some library modes invoke LLMs at runtime via a provider abstraction (hosted APIs or local open-weights). The default `fast` runtime mode is retrieval-based — config generation happens once, offline, during dataset construction; runtime resolution is a local lookup followed by local audio rendering.

**Synthetic research artifact.** The retrieval map released with, or linked from, this repository is AI-generated: a hosted LLM produced candidate `MusicConfig`s — together with AI-generated titles, short rationales, and color palettes — from natural-language scenes drawn from an openly licensed corpus; candidates were schema-validated, rendered to audio, scored by an audio–language similarity model, and the best valid ones retained. These scores are treated as an engineering proxy, not as a claim about musical quality. Scripts, prompts, and model identifiers live in the data-processing pipeline directory. No human-subject data, private user logs, or private personal data was used in constructing this artifact.

## Representative sample of an AI-assisted coding session

The exchange below is a fictional but realistic illustration of the kind of AI-assisted coding session used during development. It is heavily compressed — each turn here stands in for many that didn't work. A representative example is shared rather than a real transcript to respect the privacy of the author.

**Author.** What runtime modes does the library currently expose, and which one is the default?

**Assistant.** The user-facing default is `fast`, surfaced through `latentscore.render(...)`, `latentscore.stream(...)`, and `latentscore.live(...)`:

- `fast` (default): local embedding retrieval over a precomputed prompt-to-config map; no LLM call at runtime.
- `fast_heavy`: optional CLAP-audio embedding retrieval; requires the `[heavy]` extra and a CLAP embedding map.
- `external:<model>` or `ExternalModelSpec(...)`: hosted LLM path via the LiteLLM adapter; requires `[external]`.
- `expressive` / `local`: local open-weights model path; requires `[expressive]` and downloaded model assets.
- direct `MusicConfig` / `MusicConfigUpdate`: bypasses text resolution and renders or updates a typed config directly.

*Human action:* confirmed against `latentscore/models.py` and `latentscore/dx.py` before continuing.

**Author.** I need an LLM-facing schema in `latentscore/config.py` that uses string labels for fields the synthesizer wants as floats (e.g. `swing`, `melody_density`, `syncopation`), with `extra="forbid"`, and a method that produces a public `MusicConfig`. The public `MusicConfig` should keep defaults and tolerate extras.

**Assistant.**

```python
class MusicConfigPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tempo: TempoLabel = Field(description=_PROMPT_DESC["tempo"])
    swing: SwingLabel = Field(description=_PROMPT_DESC["swing"])
    melody_density: MelodyDensityLabel = Field(description=_PROMPT_DESC["melody_density"])
    # …remaining fields…

    def to_config(self) -> "MusicConfig":
        data = self.model_dump()
        data["swing"] = swing_to_float(self.swing)
        data["melody_density"] = melody_density_to_float(self.melody_density)
        # …remaining label→float conversions…
        return MusicConfig.model_validate(data)
```

`MusicConfig` itself uses `extra="allow"` and ships defaults for every field. *Human action:* adopted; aligned field order with the existing module.

**Author.** Factor the label→float conversion into named per-field helpers (`swing_to_float`, etc.) backed by frozen mappings, so invalid labels raise a clear domain error rather than a `KeyError`.

**Assistant.**

```python
_SWING_MAP: Mapping[SwingLabel, float] = MappingProxyType(
    {"none": 0.0, "light": 0.2, "medium": 0.5, "heavy": 0.8}
)

def swing_to_float(value: SwingLabel) -> float:
    try:
        return _SWING_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown swing label: {value!r}") from exc
```

Same pattern for `melody_density_to_float`, `syncopation_to_float`, etc. Each map is typed `Mapping[<Literal>, float]`. *Human action:* adopted unchanged.

**Author.** The LLM returns more than just a config — it also produces a title, a short rationale, and three color palettes. Add a payload wrapper for the full response.

**Assistant.**

```python
class MusicConfigPromptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thinking: str = Field(..., max_length=MAX_LONG_FIELD_CHARS)
    title: str = Field(..., min_length=1, max_length=MAX_TITLE_CHARS)
    config: MusicConfigPrompt
    palettes: list[Palette] = Field(..., min_length=3, max_length=3)
```

Added a `@field_validator` on `title` rejecting empty/whitespace-only strings and titles over `MAX_TITLE_WORDS` words. *Human action:* adopted with a tighter check on whitespace.

**Author.** `MusicConfigPrompt` and `MusicConfig` need to stay structurally aligned — adding a field to one and forgetting the other should fail loudly. Add an import-time parity check.

**Assistant.**

```python
def _assert_prompt_schema_parity() -> None:
    prompt_fields = set(MusicConfigPrompt.model_fields)
    config_fields = set(MusicConfig.model_fields)
    excluded = {"schema_version"}
    if prompt_fields == (config_fields - excluded):
        return
    missing = sorted(config_fields - prompt_fields - excluded)
    extra = sorted(prompt_fields - config_fields)
    raise AssertionError(
        f"MusicConfigPrompt mismatch: missing={missing!r}, extra={extra!r}"
    )

_assert_prompt_schema_parity()
```

Compares `model_fields` sets between the two schemas, excluding fields like `schema_version` that intentionally live only on the runtime config. Runs at module import so any drift fails at first use. *Human action:* adopted.

**Author.** Write a parametrized test for `swing_to_float` covering every valid label plus an invalid one, and run it in the project env.

**Assistant.**

```python
# tests/test_latentscore_config.py
@pytest.mark.parametrize(
    "label,expected",
    [("none", 0.0), ("light", 0.2), ("medium", 0.5), ("heavy", 0.8)],
)
def test_swing_to_float_valid(label, expected):
    assert swing_to_float(label) == expected

def test_swing_to_float_invalid():
    with pytest.raises(InvalidConfigError):
        swing_to_float("galloping")
```

Ran the suite with `pytest tests/test_latentscore_config.py -q`; it passed.

*Human action:* adopted; extended coverage to the other per-field helpers via a single parametrized fixture iterating each `_*_MAP`.