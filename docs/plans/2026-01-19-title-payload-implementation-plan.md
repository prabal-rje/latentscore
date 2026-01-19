# Title Payload Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a required `title` field to `MusicConfigPromptPayload`, update prompts and reward shaping, and validate the new field across generation, training, inference, and evaluation.

**Architecture:** Extend the shared payload schema in `common/music_schema.py` and `latentscore/config.py`, update prompt templates to require `title` between `thinking` and `config`, and incorporate title similarity + length penalty into GRPO rewards. Update fixtures/tests/docs and run tiny pipeline runs for verification.

**Tech Stack:** Python 3.10, Pydantic v2, pytest, ruff, litellm, sentence-transformers (optional).

### Task 1: Schema + Prompt Updates (TDD)

**Files:**
- Modify: `common/music_schema.py`
- Modify: `latentscore/config.py`
- Modify: `common/prompts.py`
- Modify: `tests/data_work/test_music_schema.py`
- Modify: `tests/data_work/test_music_prompt.py`
- Modify: `tests/test_latentscore_prompt_schema.py`

**Step 1: Write the failing test**

```python
def test_music_schema_requires_title() -> None:
    payload = {"thinking": "x", "config": {}, "palettes": []}
    with pytest.raises(ValidationError):
        MusicConfigPromptPayload.model_validate(payload)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data_work/test_music_schema.py::test_music_schema_requires_title -v`
Expected: FAIL because title is not required yet.

**Step 3: Write minimal implementation**

```python
# common/music_schema.py
MAX_TITLE_CHARS = 60
MAX_TITLE_WORDS = 6

class MusicConfigPromptPayload(BaseModel):
    thinking: str = Field(...)
    title: str = Field(..., max_length=MAX_TITLE_CHARS, description=PROMPT_DESC["title"])
    config: MusicConfigPrompt = Field(...)
    palettes: list[Palette] = Field(...)

    @field_validator("title")
    def _validate_title(cls, value: str) -> str:
        words = [word for word in value.split() if word]
        if not words:
            raise ValueError("title must not be empty")
        if len(words) > MAX_TITLE_WORDS:
            raise ValueError("title exceeds max word count")
        return value
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data_work/test_music_schema.py::test_music_schema_requires_title -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add common/music_schema.py latentscore/config.py common/prompts.py tests/data_work/test_music_schema.py tests/data_work/test_music_prompt.py tests/test_latentscore_prompt_schema.py
git commit -m "feat: add title field to prompt payload"
```

### Task 2: Reward Shaping for Title (TDD)

**Files:**
- Modify: `common/reward_config.py`
- Modify: `data_work/lib/rewards.py`
- Modify: `data_work/03_modal_train.py`
- Modify: `tests/data_work/test_rewards.py`

**Step 1: Write the failing test**

```python
def test_title_similarity_and_length_penalty() -> None:
    payload = {"thinking": "ok", "title": "neon rain city", "config": {...}, "palettes": []}
    result = compute_partial_reward("neon rain", json.dumps(payload))
    assert result.title_similarity is not None
    assert result.title_length_penalty == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data_work/test_rewards.py::test_title_similarity_and_length_penalty -v`
Expected: FAIL (missing fields/logic).

**Step 3: Write minimal implementation**

```python
# data_work/lib/rewards.py
# add title_similarity, title_length_penalty, title_score to RewardBreakdown
# compute similarity (sentence-transformers if available, fallback to lexical)
# apply linear penalty if title length exceeds MAX_TITLE_CHARS
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data_work/test_rewards.py::test_title_similarity_and_length_penalty -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add common/reward_config.py data_work/lib/rewards.py data_work/03_modal_train.py tests/data_work/test_rewards.py
git commit -m "feat: add title reward shaping"
```

### Task 3: Update Payload Producers + Examples

**Files:**
- Modify: `data_work/lib/baselines.py`
- Modify: `data_work/02b_generate_configs.py`
- Modify: `latentscore/prompt_examples.py`
- Modify: `tests/test_litellm_adapter.py`
- Modify: `tests/test_models_prompt.py`

**Step 1: Update payload constructors to include title**

```python
MusicConfigPromptPayload(thinking=..., title=..., config=..., palettes=...)
```

**Step 2: Update prompt examples with short titles (<=6 words)**

**Step 3: Run focused tests**

Run: `pytest tests/test_litellm_adapter.py::test_litellm_adapter_enforces_json -v`
Expected: PASS.

**Step 4: Commit**

```bash
git add data_work/lib/baselines.py data_work/02b_generate_configs.py latentscore/prompt_examples.py tests/test_litellm_adapter.py tests/test_models_prompt.py
git commit -m "feat: add title to payload producers"
```

### Task 4: Update Fixtures, Evaluation, Docs

**Files:**
- Modify: `data_work/.smoke/sft_smoke.jsonl`
- Modify: `data_work/lib/eval_schema.py`
- Modify: `data_work/README.md`
- Modify: `data_work/METHODOLOGY.md`
- Modify: `data_work/EXPERIMENTS.md`
- Modify: `tests/data_work/test_generate_configs_roles.py`
- Modify: `tests/data_work/test_generate_configs_batching.py`
- Modify: `tests/data_work/test_generate_configs_types.py`
- Modify: `tests/data_work/test_score_configs.py`
- Modify: `tests/data_work/test_config_batcher.py`

**Step 1: Update fixture payloads to include `title`**

**Step 2: Exclude `title` from field distributions**

**Step 3: Update docs to mention `title` in payload schema**

**Step 4: Run focused tests**

Run: `pytest tests/data_work/test_generate_configs_roles.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add data_work/.smoke/sft_smoke.jsonl data_work/lib/eval_schema.py data_work/README.md data_work/METHODOLOGY.md data_work/EXPERIMENTS.md tests/data_work
git commit -m "chore: update fixtures and docs for title"
```

### Task 5: Tiny Pipeline Verification Runs

**Files:**
- Modify: `data_work/02a_extract_vibes.py` (only if temporary logging needed)
- Modify: `data_work/02b_generate_configs.py` (only if temporary logging needed)
- Modify: `data_work/02c_score_configs.py` (only if temporary logging needed)
- Modify: `data_work/04_clap_benchmark.py` (only if temporary logging needed)

**Step 1: Run 02a with tiny config**
Run: `conda run -n latentscore-data python data_work/02a_extract_vibes.py --config data_work/.tmp_configs/run_config.json`

**Step 2: Run 02b with tiny config**
Run: `conda run -n latentscore-data python data_work/02b_generate_configs.py --config data_work/.tmp_configs/run_config.json`

**Step 3: Run 02c with tiny config**
Run: `conda run -n latentscore-data python data_work/02c_score_configs.py --config data_work/.tmp_configs/run_config.json`

**Step 4: Run 04 benchmark with tiny config**
Run: `conda run -n latentscore-data python data_work/04_clap_benchmark.py --config data_work/.tmp_configs/run_config.json`

**Step 5: Verify outputs manually**
Check: `data_work/.tmp_configs` outputs include `title` and ordering.

### Task 6: Full Verification

**Step 1: Run test suite**
Run: `conda run -n latentscore-data pytest -q`
Expected: PASS.

**Step 2: Run checks**
Run: `ENV_NAME=latentscore-data make check`
Expected: PASS (pyright warnings may remain).

**Step 3: Commit any final formatting fixes**

```bash
git add -u
git commit -m "chore: finalize title payload rollout"
```
