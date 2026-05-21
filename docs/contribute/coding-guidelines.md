# Coding Guidelines

This document establishes the coding standards for this codebase. All code contributions must adhere to these guidelines.

## Table of Contents

1. [Type Safety](#type-safety)
2. [Functional Programming Style](#functional-programming-style)
3. [Control Flow](#control-flow)
4. [Error Handling](#error-handling)
5. [State Management](#state-management)
6. [Code Organization](#code-organization)
7. [Dependency Injection](#dependency-injection)
8. [Constants and Magic Values](#constants-and-magic-values)
9. [Async Programming](#async-programming)
10. [Code Auditing](#code-auditing)

---

## Type Safety

### Prefer Pydantic Over Dataclasses

Use Pydantic models by default. Only use `dataclass` when performance is critical and you've measured the overhead.

```python
# GOOD: Pydantic with strict configuration
from pydantic import BaseModel, ConfigDict

class UserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    email: str
    age: int


# BAD: Raw dataclass without good reason
from dataclasses import dataclass

@dataclass
class UserConfig:
    name: str
    email: str
    age: int
```

### Assert Before Casting

Never cast without first asserting your assumptions. Casts are lies to the type checker - verify the lie is true first.

```python
from typing import Any

# GOOD: Assert then cast
def process_config(data: Any) -> str:
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    assert "name" in data, "Missing required field 'name'"
    assert isinstance(data["name"], str), f"Expected str, got {type(data['name'])}"
    return data["name"]


# BAD: Blind casting
from typing import cast

def process_config(data: Any) -> str:
    return cast(dict[str, Any], data)["name"]  # Dangerous!


# GOOD: Using Pydantic for validation (preferred)
from pydantic import BaseModel

class Config(BaseModel):
    name: str

def process_config(data: Any) -> str:
    validated = Config.model_validate(data)  # Raises if invalid
    return validated.name
```

### Use Protocols for Structural Typing

When you need duck typing, use `Protocol` instead of `Any`.

```python
from typing import Protocol

# GOOD: Protocol defines the expected interface
class Scorer(Protocol):
    def score(self, text: str, audio_path: str) -> float:
        ...

def evaluate(scorer: Scorer, text: str, path: str) -> float:
    return scorer.score(text, path)


# BAD: Any loses all type information
from typing import Any

def evaluate(scorer: Any, text: str, path: str) -> float:
    return scorer.score(text, path)  # No type checking!
```

### Keep Type Checking Simple

Avoid over-engineered type checking. Use the simplest pattern that works.

```python
# GOOD: Simple dict() pattern for JSON data
def parse_response(data: Any) -> str:
    match data:
        case dict():
            return data.get("value", "")
        case _:
            return ""


# GOOD: Direct type pattern when you know the source
def process_json(text: str) -> dict[str, Any] | None:
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    match result:
        case dict():
            return result
        case _:
            return None


# BAD: Over-engineered with Mapping when dict() suffices
from collections.abc import Mapping

def parse_response(data: Any) -> str:
    # Mapping() doesn't work as a match pattern!
    # And JSON always produces dict, not other Mapping types
    match data:
        case Mapping():  # TypeError: called match pattern must be a type
            return data.get("value", "")
        case _:
            return ""


# BAD: Unnecessary type gymnastics
def process_json(text: str) -> Mapping[str, Any] | None:
    # Just use dict - that's what json.loads returns
    ...
```

**Key principle:** Match the type check to your actual data source. If you're parsing JSON, use `dict()` - that's what `json.loads()` returns. Don't reach for abstract types like `Mapping` unless you genuinely need to accept multiple mapping types.

### Use TypedDict for Structured Dictionaries

When you must use dictionaries with known keys, use `TypedDict`.

```python
from typing import TypedDict

# GOOD: TypedDict provides structure
class VibeRow(TypedDict):
    dataset: str
    id_in_dataset: str
    vibe_text: str
    score: float


def process_row(row: VibeRow) -> str:
    return row["vibe_text"]  # Type-checked!


# BAD: Untyped dict
def process_row(row: dict[str, Any]) -> str:
    return row["vibe_text"]  # No guarantees!
```

---

## Functional Programming Style

### Prefer map, filter, reduce Over Imperative Loops

Use functional constructs for transformations. They're more declarative and composable.

```python
from functools import reduce
from operator import add

# GOOD: Functional style
def sum_scores(items: list[dict[str, float]]) -> float:
    scores = map(lambda x: x["score"], items)
    valid_scores = filter(lambda s: s > 0, scores)
    return reduce(add, valid_scores, 0.0)


# GOOD: List comprehension for simple cases
def get_names(users: list[User]) -> list[str]:
    return [u.name for u in users if u.active]


# BAD: Imperative accumulation
def sum_scores(items: list[dict[str, float]]) -> float:
    total = 0.0
    for item in items:
        score = item["score"]
        if score > 0:
            total += score
    return total
```

### Use functools and itertools

Leverage the standard library's functional tools.

```python
from functools import partial, lru_cache, reduce
from itertools import chain, groupby, islice, takewhile
from operator import attrgetter, itemgetter

# GOOD: functools.partial for currying
def multiply(x: int, y: int) -> int:
    return x * y

double = partial(multiply, 2)
triple = partial(multiply, 3)


# GOOD: itertools for lazy iteration
def process_large_dataset(paths: list[Path]) -> Iterator[Record]:
    all_records = chain.from_iterable(map(load_jsonl, paths))
    valid_records = filter(is_valid, all_records)
    return islice(valid_records, 1000)  # First 1000 valid


# GOOD: groupby for grouping
def group_by_category(items: list[Item]) -> dict[str, list[Item]]:
    sorted_items = sorted(items, key=attrgetter("category"))
    return {
        k: list(v)
        for k, v in groupby(sorted_items, key=attrgetter("category"))
    }


# GOOD: lru_cache for memoization
@lru_cache(maxsize=128)
def expensive_computation(key: str) -> Result:
    return compute(key)
```

### Use Generators for Lazy Evaluation

Prefer generators over lists when processing large datasets.

```python
from collections.abc import Iterator
from typing import Iterable

# GOOD: Generator for lazy evaluation
def iter_valid_records(path: Path) -> Iterator[Record]:
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            if is_valid(record):
                yield Record.model_validate(record)


# GOOD: Generator expression
def get_scores(records: Iterable[Record]) -> Iterator[float]:
    return (r.score for r in records if r.score is not None)


# BAD: Materializing everything into memory
def get_all_valid_records(path: Path) -> list[Record]:
    records = []
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            if is_valid(record):
                records.append(Record.model_validate(record))
    return records  # Could be millions of records!
```

---

## Control Flow

### Use Match Statements for Type Dispatch

Prefer `match` over if/elif chains, especially for type checks.

```python
from typing import Any

# GOOD: Match statement with type patterns
def process_value(value: Any) -> str:
    match value:
        case str():
            return value.upper()
        case int() | float():
            return str(value)
        case list():
            return ", ".join(map(str, value))
        case dict():
            return json.dumps(value)
        case None:
            return ""
        case _:
            raise TypeError(f"Unsupported type: {type(value)}")


# GOOD: Match for structured data
def handle_response(response: dict[str, Any]) -> Result:
    match response:
        case {"status": "success", "data": data}:
            return Result.success(data)
        case {"status": "error", "message": msg}:
            return Result.error(msg)
        case {"status": status}:
            return Result.error(f"Unknown status: {status}")
        case _:
            return Result.error("Invalid response format")


# BAD: If/elif chain
def process_value(value: Any) -> str:
    if isinstance(value, str):
        return value.upper()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        return ", ".join(map(str, value))
    elif isinstance(value, dict):
        return json.dumps(value)
    elif value is None:
        return ""
    else:
        raise TypeError(f"Unsupported type: {type(value)}")
```

### Use Early Returns

Return early to avoid deep nesting and make the happy path clear.

```python
# GOOD: Early returns
def process_record(record: dict[str, Any] | None) -> Result:
    if record is None:
        return Result.error("Record is None")

    if "id" not in record:
        return Result.error("Missing id field")

    if not isinstance(record["id"], str):
        return Result.error("id must be a string")

    # Happy path - no nesting
    return Result.success(transform(record))


# BAD: Deep nesting
def process_record(record: dict[str, Any] | None) -> Result:
    if record is not None:
        if "id" in record:
            if isinstance(record["id"], str):
                return Result.success(transform(record))
            else:
                return Result.error("id must be a string")
        else:
            return Result.error("Missing id field")
    else:
        return Result.error("Record is None")
```

---

## Error Handling

### Never Silently Pass Exceptions

Every caught exception must be logged or re-raised. Silent `pass` is forbidden.

```python
import logging

LOGGER = logging.getLogger(__name__)

# GOOD: Log and handle
def safe_parse(data: str) -> dict[str, Any] | None:
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse JSON: %s", exc)
        return None


# GOOD: Log and re-raise with context
def load_config(path: Path) -> Config:
    try:
        data = json.loads(path.read_text())
        return Config.model_validate(data)
    except FileNotFoundError:
        LOGGER.error("Config file not found: %s", path)
        raise
    except json.JSONDecodeError as exc:
        LOGGER.error("Invalid JSON in config file %s: %s", path, exc)
        raise ValueError(f"Config file {path} contains invalid JSON") from exc


# GOOD: Convert to Result type
def try_parse(data: str) -> Result[dict[str, Any], str]:
    try:
        return Ok(json.loads(data))
    except json.JSONDecodeError as exc:
        return Err(f"JSON parse error: {exc}")


# BAD: Silent pass - FORBIDDEN!
def safe_parse(data: str) -> dict[str, Any] | None:
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        pass  # NEVER DO THIS!
    return None


# BAD: Bare except - FORBIDDEN!
def safe_parse(data: str) -> dict[str, Any] | None:
    try:
        return json.loads(data)
    except:  # NEVER DO THIS!
        return None
```

### Specify Exception Types

Always catch specific exception types, never bare `except`.

```python
# GOOD: Specific exceptions
def fetch_data(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.Timeout:
        LOGGER.error("Request timed out: %s", url)
        raise
    except requests.HTTPError as exc:
        LOGGER.error("HTTP error %s for %s", exc.response.status_code, url)
        raise
    except requests.RequestException as exc:
        LOGGER.error("Request failed for %s: %s", url, exc)
        raise


# BAD: Catching too broadly
def fetch_data(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as exc:  # Too broad!
        LOGGER.error("Something went wrong: %s", exc)
        raise
```

---

## State Management

### Minimize Hidden State

Avoid hidden mutable state. If state is necessary, make it explicit and contained.

```python
# GOOD: Explicit state in a class with clear boundaries
class Scorer:
    def __init__(self, model_path: str) -> None:
        self._model = load_model(model_path)  # State is clear
        self._cache: dict[str, float] = {}    # Cache is explicit

    def score(self, text: str) -> float:
        if text in self._cache:
            return self._cache[text]
        result = self._model.predict(text)
        self._cache[text] = result
        return result


# GOOD: Pure functions with no hidden state
def compute_score(text: str, model: Model) -> float:
    return model.predict(text)


# GOOD: State passed explicitly
def process_batch(
    items: list[Item],
    cache: dict[str, Result],  # State is explicit parameter
) -> tuple[list[Result], dict[str, Result]]:
    results = []
    for item in items:
        if item.id in cache:
            results.append(cache[item.id])
        else:
            result = compute(item)
            cache[item.id] = result
            results.append(result)
    return results, cache


# BAD: Hidden module-level mutable state
_GLOBAL_CACHE: dict[str, float] = {}  # Hidden state!

def score(text: str) -> float:
    if text in _GLOBAL_CACHE:
        return _GLOBAL_CACHE[text]
    result = expensive_computation(text)
    _GLOBAL_CACHE[text] = result  # Side effect!
    return result


# BAD: Hidden state via closure
def make_counter():
    count = 0  # Hidden state
    def increment():
        nonlocal count
        count += 1  # Mutation!
        return count
    return increment
```

### Prefer Immutable Data

Use frozen Pydantic models and avoid mutation.

```python
from pydantic import BaseModel, ConfigDict

# GOOD: Frozen (immutable) model
class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: int

# Create new instance instead of mutating
def update_value(config: Config, new_value: int) -> Config:
    return config.model_copy(update={"value": new_value})


# BAD: Mutable state
class Config:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def set_value(self, value: int) -> None:
        self.value = value  # Mutation!
```

### State Must Have Clear Ownership

If state is necessary, there must be a single clear owner.

```python
# GOOD: Clear ownership - state lives in one place
class Pipeline:
    def __init__(self) -> None:
        self._results: list[Result] = []

    def process(self, item: Item) -> None:
        result = transform(item)
        self._results.append(result)

    def get_results(self) -> list[Result]:
        return list(self._results)  # Return copy to prevent external mutation


# BAD: Shared mutable state with unclear ownership
results: list[Result] = []

def process_a(item: Item) -> None:
    results.append(transform_a(item))

def process_b(item: Item) -> None:
    results.append(transform_b(item))

# Who owns `results`? Who clears it? Race conditions?
```

---

## Code Organization

### Nested Functions Are OK

Use nested functions to avoid polluting module namespace and to encapsulate helper logic.

```python
# GOOD: Nested helper function
def process_records(records: list[Record]) -> list[Result]:
    def transform_single(record: Record) -> Result:
        """Transform a single record - only used here."""
        validated = validate(record)
        enriched = enrich(validated)
        return Result.from_record(enriched)

    return [transform_single(r) for r in records]


# GOOD: Nested function for complex key
def group_records(records: list[Record]) -> dict[str, list[Record]]:
    def make_key(record: Record) -> str:
        return f"{record.dataset}:{record.category}"

    result: dict[str, list[Record]] = {}
    for record in records:
        key = make_key(record)
        result.setdefault(key, []).append(record)
    return result


# BAD: Polluting module namespace with one-off helpers
def _transform_single_record_for_process_records(record: Record) -> Result:
    """Only used by process_records but clutters module."""
    validated = validate(record)
    enriched = enrich(validated)
    return Result.from_record(enriched)

def process_records(records: list[Record]) -> list[Result]:
    return [_transform_single_record_for_process_records(r) for r in records]
```

---

## Dependency Injection

### Inject Dependencies for Testability

Pass dependencies as parameters instead of hardcoding them. This makes testing trivial.

```python
from typing import Protocol

# GOOD: Dependencies injected via Protocol
class HttpClient(Protocol):
    def get(self, url: str) -> bytes:
        ...

class DataFetcher:
    def __init__(self, client: HttpClient) -> None:
        self._client = client

    def fetch_config(self, url: str) -> Config:
        data = self._client.get(url)
        return Config.model_validate_json(data)


# Easy to test with a mock!
class MockHttpClient:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self._responses = responses

    def get(self, url: str) -> bytes:
        return self._responses[url]

def test_fetch_config():
    mock_client = MockHttpClient({
        "http://example.com/config": b'{"name": "test", "value": 42}'
    })
    fetcher = DataFetcher(mock_client)
    config = fetcher.fetch_config("http://example.com/config")
    assert config.name == "test"


# BAD: Hardcoded dependency - hard to test!
import requests

class DataFetcher:
    def fetch_config(self, url: str) -> Config:
        response = requests.get(url)  # Hardcoded!
        return Config.model_validate_json(response.content)
```

### Use Factory Functions for Complex Setup

```python
# GOOD: Factory with dependency injection
def create_pipeline(
    *,
    scorer: Scorer,
    cache: Cache,
    logger: Logger,
) -> Pipeline:
    return Pipeline(scorer=scorer, cache=cache, logger=logger)


# In production
pipeline = create_pipeline(
    scorer=ClapScorer(),
    cache=RedisCache(host="localhost"),
    logger=get_logger(__name__),
)

# In tests
pipeline = create_pipeline(
    scorer=MockScorer(return_value=0.5),
    cache=InMemoryCache(),
    logger=NullLogger(),
)
```

---

## Constants and Magic Values

### No Magic Constants

Never use unexplained literal values. Define named constants.

```python
# GOOD: Named constants with clear meaning
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 3
MIN_SCORE_THRESHOLD = 0.5
BATCH_SIZE = 100

def fetch_with_retry(url: str) -> bytes:
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            return fetch(url, timeout=DEFAULT_TIMEOUT_SECONDS)
        except TimeoutError:
            if attempt == MAX_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)


# GOOD: Constants scoped to function if only used there
def process_audio(path: Path) -> Result:
    _SAMPLE_RATE = 44100
    _CHANNELS = 2
    _BIT_DEPTH = 16

    audio = load_audio(path, sample_rate=_SAMPLE_RATE)
    return process(audio, channels=_CHANNELS, bit_depth=_BIT_DEPTH)


# BAD: Magic numbers
def fetch_with_retry(url: str) -> bytes:
    for attempt in range(3):  # What is 3?
        try:
            return fetch(url, timeout=30)  # What is 30?
        except TimeoutError:
            if attempt == 2:  # Why 2?
                raise
            time.sleep(2 ** attempt)
```

### Use Underscores for Internal Constants

```python
# Module-level internal constants use underscore prefix
_DEFAULT_MODEL_PATH = Path("models/default.bin")
_SUPPORTED_FORMATS = frozenset({"mp3", "wav", "flac"})

# Public constants (part of module API) have no prefix
SAMPLE_RATE = 44100
MAX_DURATION = 300.0
```

---

## Async Programming

### Use asyncio When Necessary

Use async for I/O-bound operations, especially when dealing with multiple concurrent operations.

```python
import asyncio
from collections.abc import Sequence

# GOOD: Async for concurrent I/O
async def fetch_all_configs(urls: Sequence[str]) -> list[Config]:
    async def fetch_one(url: str) -> Config:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                return Config.model_validate(data)

    tasks = [fetch_one(url) for url in urls]
    return await asyncio.gather(*tasks)


# GOOD: Semaphore for rate limiting
async def fetch_with_limit(
    urls: Sequence[str],
    max_concurrent: int = 10,
) -> list[Config]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str) -> Config:
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    return Config.model_validate(data)

    tasks = [fetch_one(url) for url in urls]
    return await asyncio.gather(*tasks)


# BAD: Sync I/O in a loop (slow!)
def fetch_all_configs(urls: Sequence[str]) -> list[Config]:
    results = []
    for url in urls:
        response = requests.get(url)  # Blocking!
        results.append(Config.model_validate(response.json()))
    return results
```

### Don't Mix Sync and Async Carelessly

```python
# GOOD: Async all the way down
async def process_pipeline(items: list[Item]) -> list[Result]:
    async def process_one(item: Item) -> Result:
        data = await fetch_data(item.url)
        return await transform(data)

    return await asyncio.gather(*[process_one(i) for i in items])


# BAD: Blocking call in async context
async def process_pipeline(items: list[Item]) -> list[Result]:
    results = []
    for item in items:
        data = requests.get(item.url)  # BLOCKING in async context!
        results.append(transform(data))
    return results
```

---

## Code Auditing

### Read the Code, Don't Just Grep

When auditing code for style violations, **actually read and understand each file**. Don't rely on pattern matching or grep searches to find issues.

```bash
# BAD: Grepping for patterns
grep -r "isinstance" data_work/  # Finds matches but no context
grep -r "\.get(" data_work/      # Too noisy, false positives

# BAD: Assuming patterns mean violations
# Not every isinstance() is wrong
# Not every .get() needs to be a TypedDict
# Not every dict is a Pydantic model waiting to happen
```

**The right approach:**

1. **Read each file completely** - understand what it does
2. **Identify actual violations** - not pattern matches, but genuine style issues
3. **Consider context** - some patterns are appropriate in certain contexts
4. **Compile a list** - document each violation with file, line, and reason
5. **Fix systematically** - work through the list, verify each fix

### Understand When Patterns Are Appropriate

```python
# isinstance() is APPROPRIATE here - checking external data type
def process_json_value(value: Any) -> str:
    match value:
        case str():   # This is the right way
            return value
        case _:
            return str(value)


# isinstance() would be WRONG here - should use Pydantic
def validate_config(data: dict[str, Any]) -> bool:
    # BAD: Manual validation
    if not isinstance(data.get("name"), str):
        return False
    if not isinstance(data.get("count"), int):
        return False
    return True

    # GOOD: Use Pydantic instead
    # try:
    #     Config.model_validate(data)
    #     return True
    # except ValidationError:
    #     return False


# .get() is APPROPRIATE for optional dict access
def extract_optional(data: dict[str, Any]) -> str | None:
    return data.get("optional_field")  # Fine - explicit optional


# .get() is WRONG when you should have typed data
def process_config(config: dict[str, Any]) -> str:
    # BAD: Accessing known fields from untyped dict
    return config.get("name", "")

    # GOOD: Use TypedDict or Pydantic
    # config: Config
    # return config.name
```

### Handle Dead Code Carefully

When you suspect code is dead (unused files, functions, modules), **don't delete it immediately**. Ask the user first.

**The process:**

1. **Identify potentially dead code** - unused imports, unreferenced functions, orphaned files
2. **Ask the user** - "Is `lib/old_scorer.py` dead code that should be removed?"
3. **If confirmed dead** - rename to `_<filename>` to "un-importantize" it
4. **User removes explicitly** - the user deletes when they're confident

```bash
# GOOD: Rename to mark as probably-dead
mv data_work/lib/old_scorer.py data_work/lib/_old_scorer.py
mv data_work/lib/unused_utils.py data_work/lib/_unused_utils.py

# The underscore prefix:
# - Signals "this is probably dead"
# - Keeps the file available if needed
# - Makes it obvious in directory listings
# - User can grep for "_" prefixed files to clean up later
```

**Why this approach:**

- **Safe** - no accidental deletion of code that's actually used elsewhere
- **Visible** - dead code is marked, not hidden
- **Reversible** - easy to restore if the code turns out to be needed
- **User-controlled** - final deletion is an explicit user action

```python
# When auditing, compile a dead code report:

## Potentially Dead Code

### Files
- [ ] `lib/old_scorer.py` - no imports found, last modified 6 months ago
- [ ] `lib/legacy_utils.py` - only imported by deleted test file

### Functions
- [ ] `rewards.py:_palette_is_valid()` - defined but never called
- [ ] `llm_client.py:_legacy_parse()` - commented "TODO: remove"

### Ask user before proceeding with renames.
```

### Document Your Audit

When auditing, create a structured report:

```markdown
## Style Audit: data_work/

### File: rewards.py

| Line | Issue | Current | Should Be |
|------|-------|---------|-----------|
| 154 | isinstance cascade | `if isinstance(x, dict)` | `match x: case dict()` |
| 225 | raw dict access | `palettes.get("colors")` | TypedDict or Pydantic |

### File: jsonl_io.py

| Line | Issue | Current | Should Be |
|------|-------|---------|-----------|
| 17 | isinstance check | `if isinstance(row, dict)` | `match row: case dict()` |

### Exceptions (Appropriate Usage)

- `llm_client.py:150` - isinstance in `_parse_json_payload` is correct (external JSON data)
- `config_io.py:42` - .get() on truly dynamic config data is appropriate
```

This structured approach ensures you understand the code, distinguish real violations from appropriate patterns, and can track your fixes systematically.

---

## Summary Checklist

Before submitting code, verify:

- [ ] All types are explicit (no untyped `Any` without good reason)
- [ ] Pydantic models used for data structures (not raw dicts)
- [ ] Assertions precede any type casts
- [ ] Type checks are simple (use `dict()` for JSON, not `Mapping()`)
- [ ] Functional style used where appropriate (map/filter/reduce)
- [ ] `match` statements used for type dispatch
- [ ] Early returns used to avoid nesting
- [ ] No silent exception handling (no bare `pass`)
- [ ] No hidden mutable state (state ownership is clear)
- [ ] Dependencies are injected (not hardcoded)
- [ ] No magic constants (all values are named)
- [ ] Async used for concurrent I/O operations

**For code reviews/audits:**

- [ ] Read each file completely (don't just grep for patterns)
- [ ] Distinguish genuine violations from appropriate usage
- [ ] Document findings with file, line, and reason
- [ ] Consider context before flagging as violation
- [ ] Ask user about suspected dead code before removing
- [ ] Rename dead code to `_<filename>` instead of deleting
