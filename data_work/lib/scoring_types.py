"""Strict typing for scoring results across all data_work scoring systems.

All scoring results MUST implement the ScoreResult protocol, which requires:
- A `final_score` property returning a float in [0.0, 1.0]

This ensures consistent interfaces across:
- CLAP scoring (clap_scorer.py)
- LLM-as-judge scoring (llm_scorer.py)
- Reward computation (rewards.py)
- Evaluation results (eval_schema.py)
- Config generation scoring (02b, 02c scripts)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class ScoreResult(Protocol):
    """Protocol that all scoring results must implement.

    Any scoring result - whether from CLAP, LLM judge, reward computation,
    or custom scorers - MUST provide a final_score property.

    The final_score should be normalized to [0.0, 1.0] where:
    - 0.0 = worst possible score
    - 1.0 = best possible score

    Example implementation:
        class MyScore(BaseModel):
            internal_metric_a: float
            internal_metric_b: float

            @property
            def final_score(self) -> float:
                return (self.internal_metric_a + self.internal_metric_b) / 2.0
    """

    @property
    @abstractmethod
    def final_score(self) -> float:
        """Return the final normalized score in [0.0, 1.0]."""
        ...


# Type variable for generic score result handling
T_ScoreResult = TypeVar("T_ScoreResult", bound=ScoreResult)


def validate_score_result(result: Any, source: str = "scorer") -> None:
    """Validate that a result implements ScoreResult protocol.

    Args:
        result: The scoring result to validate
        source: Name of the scorer for error messages

    Raises:
        TypeError: If result doesn't implement ScoreResult protocol
        ValueError: If final_score is not in valid range
    """
    if not isinstance(result, ScoreResult):
        raise TypeError(
            f"Scorer '{source}' returned {type(result).__name__} which does not "
            f"implement ScoreResult protocol. Must have 'final_score' property."
        )

    score = result.final_score
    if not isinstance(score, (int, float)):
        raise TypeError(
            f"Scorer '{source}' final_score must be numeric, got {type(score).__name__}"
        )

    if not (0.0 <= score <= 1.0):
        raise ValueError(
            f"Scorer '{source}' final_score must be in [0.0, 1.0], got {score}"
        )


def score_result_to_dict(result: ScoreResult) -> dict[str, Any]:
    """Convert a ScoreResult to a dict, ensuring final_score is included.

    Works with Pydantic models (via model_dump) or any object with __dict__.

    Args:
        result: A ScoreResult instance

    Returns:
        Dict containing all fields plus guaranteed final_score
    """
    # Try Pydantic model_dump first
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif hasattr(result, "__dict__"):
        data = dict(result.__dict__)
    else:
        data = {}

    # Ensure final_score is present (it's a property, may not be in model_dump)
    data["final_score"] = result.final_score
    return data


class DictScoreResult:
    """Wrapper to make a dict implement ScoreResult protocol.

    Use this when you have a dict with a final_score key and need
    to pass it to functions expecting ScoreResult.

    Example:
        raw_scores = {"metric_a": 0.8, "metric_b": 0.6, "final_score": 0.7}
        result = DictScoreResult(raw_scores)
        validate_score_result(result)  # OK
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize with a dict containing final_score.

        Args:
            data: Dict that must contain "final_score" key

        Raises:
            KeyError: If "final_score" not in data
        """
        if "final_score" not in data:
            raise KeyError("Dict must contain 'final_score' key to be a valid ScoreResult")
        self._data = data

    @property
    def final_score(self) -> float:
        """Return the final score from the wrapped dict."""
        return float(self._data["final_score"])

    def to_dict(self) -> dict[str, Any]:
        """Return the underlying dict."""
        return dict(self._data)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Allow dict-like get access."""
        return self._data.get(key, default)
