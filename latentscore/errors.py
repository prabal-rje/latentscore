from __future__ import annotations


class LatentScoreError(Exception):
    """Base error for the LatentScore library."""


class InvalidConfigError(LatentScoreError):
    """Raised when a config cannot be parsed or validated."""


class LLMInferenceError(LatentScoreError):
    """Raised when a model provider fails to produce a response."""


class ConfigGenerateError(LatentScoreError):
    """Raised when a model fails while generating a config."""


class ModelNotAvailableError(LatentScoreError):
    """Raised when required model weights or dependencies are missing."""
