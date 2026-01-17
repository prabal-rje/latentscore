# Evaluation Sets

Fixed evaluation sets for benchmarking vibe-to-config models.

## Directory Structure

```
eval_sets/
├── short_prompts.jsonl          # 30 general short vibes
├── controllability/
│   ├── tempo_prompts.jsonl      # 20 tempo-focused prompts
│   ├── mode_prompts.jsonl       # 20 mode-focused prompts
│   └── brightness_prompts.jsonl # 20 brightness-focused prompts
└── README.md
```

## Schema

Each JSONL file contains `EvalPrompt` records (see `data_work/lib/eval_schema.py`):

```json
{
  "id": "short_001",
  "prompt": "lonely mass, drifting through the void",
  "category": "short",
  "subcategory": null,
  "expected_fields": {},
  "difficulty": "easy",
  "source": "manual",
  "paraphrase_group": null,
  "notes": "Classic ambient vibe"
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `prompt` | string | The vibe text to evaluate |
| `category` | enum | `short`, `long`, `ood`, `typo`, `controllability` |
| `subcategory` | enum? | For controllability: `tempo`, `mode`, `brightness`, etc. |
| `expected_fields` | dict | Expected config values, e.g., `{"tempo": "slow"}` |
| `difficulty` | enum | `easy`, `medium`, `hard` |
| `source` | enum | `manual`, `llm_generated`, `extracted`, `corrupted` |
| `paraphrase_group` | string? | Group ID for paraphrased variants |
| `notes` | string? | Optional annotations |

## Eval Set Descriptions

### `short_prompts.jsonl`

30 general-purpose short vibes (1-5 words each) covering diverse moods:
- Ambient/ethereal (lonely, drifting, deep space)
- Urban/industrial (neon, mechanical, underground)
- Nature (mist, waves, rain)
- Emotional (melancholy, joy, tension)

No expected_fields - used for general quality assessment.

### `controllability/tempo_prompts.jsonl`

20 prompts specifically designed to test tempo control:
- Expected values: `very_slow`, `slow`, `medium`, `fast`, `very_fast`
- Includes musical tempo terms (adagio, allegro, presto, andante)
- Includes paraphrase groups (same intent, different wording)

### `controllability/mode_prompts.jsonl`

20 prompts specifically designed to test mode/key control:
- Expected values: `major`, `minor`, `dorian`, `mixolydian`
- Tests emotional associations (happy→major, sad→minor)
- Tests genre associations (jazz→dorian, folk→mixolydian)

### `controllability/brightness_prompts.jsonl`

20 prompts specifically designed to test brightness control:
- Expected values: `very_dark`, `dark`, `medium`, `bright`, `very_bright`
- Tests lighting descriptions (pitch black, dim, sunny, blinding)
- Includes paraphrase groups

## Usage

```python
from data_work.lib.jsonl_io import iter_jsonl
from data_work.lib.eval_schema import EvalPrompt

# Load an eval set
prompts = [EvalPrompt(**record) for record in iter_jsonl("data_work/eval_sets/short_prompts.jsonl")]

# Filter by category
controllability = [p for p in prompts if p.category == "controllability"]

# Filter by expected field
tempo_prompts = [p for p in prompts if "tempo" in p.expected_fields]
```

## Adding New Eval Sets

1. Create a new JSONL file following the schema
2. Use unique IDs (prefix with category, e.g., `ood_001`)
3. Set appropriate `source` field
4. For controllability tests, always specify `expected_fields`
5. Group paraphrases with `paraphrase_group`

## Future Eval Sets (TODO)

- `long_prompts.jsonl` - Multi-sentence/paragraph vibes
- `ood_prompts.jsonl` - Out-of-domain (poetry, technical, dialog)
- `typo_prompts.jsonl` - Corrupted versions of clean prompts
- `controllability/rhythm_prompts.jsonl`
- `controllability/texture_prompts.jsonl`
