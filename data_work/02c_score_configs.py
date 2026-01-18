"""Score generated configs using multiple scoring methods.

This script:
1. Reads config JSONL from 02b_generate_configs output
2. Scores each config using specified scorers (CLAP, LLM judge, custom)
3. Each scorer returns internal scores + a mandatory final_score
4. Writes results incrementally with all scores attached

Usage:
    python -m data_work.02c_score_configs \
        --input-dir data_work/.processed \
        --output-dir data_work/.scored \
        --scorers clap,llm_judge,"./my_scorer.py:score_fn"

Scorer types:
- clap: LAION-CLAP audio-text similarity (requires audio synthesis)
- llm_judge: LLM-as-judge using multimodal models (requires audio + LLM call)
- "path/to/script.py:fn_name": Custom scorer function

Custom scorers must have signature:
    def score_fn(vibe: str, config: dict) -> dict
    # Must return dict with at least {"final_score": float}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Protocol

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_work.lib.jsonl_io import iter_jsonl
from data_work.lib.llm_client import load_env_file
from data_work.lib.periodic_writer import SyncPeriodicWriter
from data_work.lib.scoring_types import DictScoreResult, validate_score_result

_LOGGER = logging.getLogger("data_work.score_configs")

DEFAULT_CACHE_PATH = Path("data_work/.cache/score_cache.sqlite")
DEFAULT_AUDIO_DURATION = 12.0
DEFAULT_PERIODIC_WRITE_INTERVAL = 60.0
RUN_CONFIG_NAME = "run_config.json"


class ScorerProtocol(Protocol):
    """Protocol for config scorers."""

    def score(self, vibe: str, config: dict[str, Any]) -> dict[str, Any]:
        """Score a config against a vibe.

        Args:
            vibe: The vibe text
            config: The config dict (MusicConfigPromptPayload format)

        Returns:
            Dict with internal scores and mandatory "final_score" (float 0-1)
        """
        ...


class ClapConfigScorer:
    """Score configs using CLAP audio-text similarity."""

    def __init__(self, duration: float = DEFAULT_AUDIO_DURATION) -> None:
        self._duration = duration
        self._scorer: Any = None

    def _ensure_scorer(self) -> Any:
        """Lazy-load CLAP scorer."""
        if self._scorer is None:
            from data_work.lib.clap_scorer import ClapScorer

            self._scorer = ClapScorer()
        return self._scorer

    def score(self, vibe: str, config: dict[str, Any]) -> dict[str, Any]:
        """Score config using CLAP."""
        from data_work.lib.clap_scorer import score_config

        scorer = self._ensure_scorer()
        result = score_config(
            vibe=vibe,
            config=config,
            scorer=scorer,
            duration=self._duration,
        )
        return {
            "audio_text_similarity": result.audio_text_similarity,
            "audio_bad_similarity": result.audio_bad_similarity,
            "excess_badness": result.excess_badness,
            "penalty": result.penalty,
            "raw_score": result.raw_score,
            "final_score": result.final_reward,
        }


class LLMJudgeScorer:
    """Score configs using LLM-as-judge with multimodal models."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash",
        api_key: str | None = None,
        duration: float = DEFAULT_AUDIO_DURATION,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._duration = duration
        self._scorer: Any = None

    def _ensure_scorer(self) -> Any:
        """Lazy-load LLM scorer."""
        if self._scorer is None:
            from data_work.lib.llm_scorer import LLMScorer

            self._scorer = LLMScorer(
                model=self._model,
                api_key=self._api_key,
            )
        return self._scorer

    def score(self, vibe: str, config: dict[str, Any]) -> dict[str, Any]:
        """Score config using LLM judge."""
        # Generate audio first
        from latentscore.audio import write_wav
        from latentscore.config import MusicConfig, MusicConfigPrompt
        from latentscore.synth import assemble

        # Extract nested config if present
        nested = config.get("config")
        config_dict = nested if isinstance(nested, dict) else config

        # Convert to MusicConfig
        try:
            prompt_config = MusicConfigPrompt.model_validate(config_dict)
            music_config = prompt_config.to_config()
        except Exception:
            music_config = MusicConfig.model_validate(config_dict)

        internal = music_config.to_internal()
        audio = assemble(internal, duration=self._duration)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            write_wav(audio_path, audio)

            scorer = self._ensure_scorer()
            result = scorer.score(vibe, str(audio_path))

        return {
            "audio_text_similarity": result.audio_text_similarity,
            "audio_bad_similarity": result.audio_bad_similarity,
            "excess_badness": result.excess_badness,
            "penalty": result.penalty,
            "raw_score": result.raw_score,
            "final_score": result.final_reward,
        }


def load_custom_scorer(spec: str) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Load a custom scorer from a Python file.

    Args:
        spec: Path to script and function name, e.g., "./my_scorer.py:score_fn"

    Returns:
        The scorer function
    """
    if ":" not in spec:
        raise ValueError(f"Custom scorer spec must be 'path/to/script.py:fn_name', got: {spec}")

    path_str, fn_name = spec.rsplit(":", 1)
    path = Path(path_str).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Custom scorer script not found: {path}")

    # Load module from file
    spec_obj = importlib.util.spec_from_file_location("custom_scorer", path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec_obj)
    sys.modules["custom_scorer"] = module
    spec_obj.loader.exec_module(module)

    if not hasattr(module, fn_name):
        raise AttributeError(f"Function '{fn_name}' not found in {path}")

    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"'{fn_name}' in {path} is not callable")

    return fn


def parse_scorers(
    scorer_specs: list[str],
    *,
    llm_model: str,
    llm_api_key: str | None,
    audio_duration: float,
) -> dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]]:
    """Parse scorer specifications into callable scorers.

    Args:
        scorer_specs: List of scorer specs (e.g., ["clap", "llm_judge", "./custom.py:fn"])
        llm_model: Model for LLM judge
        llm_api_key: API key for LLM judge
        audio_duration: Duration for audio synthesis

    Returns:
        Dict mapping scorer names to scorer functions
    """
    scorers: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {}

    for spec in scorer_specs:
        spec = spec.strip()
        if not spec:
            continue

        if spec == "clap":
            scorer = ClapConfigScorer(duration=audio_duration)
            scorers["clap"] = scorer.score
        elif spec == "llm_judge":
            scorer = LLMJudgeScorer(
                model=llm_model,
                api_key=llm_api_key,
                duration=audio_duration,
            )
            scorers["llm_judge"] = scorer.score
        elif spec.endswith(".py:") or ".py:" in spec:
            # Custom scorer: "path/to/script.py:fn_name"
            fn = load_custom_scorer(spec)
            # Use script basename as name
            name = Path(spec.split(":")[0]).stem
            scorers[name] = fn
        else:
            raise ValueError(f"Unknown scorer: {spec}. Use 'clap', 'llm_judge', or 'path.py:fn'")

    return scorers


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Score generated configs using multiple scoring methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing config JSONL files from 02b_generate_configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where scored JSONL files will be written.",
    )
    parser.add_argument(
        "--scorers",
        type=str,
        required=True,
        help=(
            "Comma-separated list of scorers. Built-in: clap, llm_judge. "
            "Custom: 'path/to/script.py:fn_name' (in quotes)."
        ),
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini/gemini-2.0-flash",
        help="Model for LLM judge scorer.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for LLM judge (or set via env var).",
    )
    parser.add_argument(
        "--llm-api-key-env",
        type=str,
        default="GEMINI_API_KEY",
        help="Environment variable for LLM API key.",
    )
    parser.add_argument(
        "--audio-duration",
        type=float,
        default=DEFAULT_AUDIO_DURATION,
        help="Audio duration in seconds for synthesis.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit total rows to process (0 = no limit).",
    )
    parser.add_argument(
        "--only-splits",
        nargs="*",
        default=None,
        help="Only process specific splits (e.g., TEST SFT-Val).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output (skip already processed rows).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--write-interval",
        type=float,
        default=DEFAULT_PERIODIC_WRITE_INTERVAL,
        help="Seconds between periodic output writes (default 60s).",
    )
    return parser


def row_key(row: dict[str, Any]) -> str:
    """Generate unique key for a row."""
    return f"{row.get('dataset')}:{row.get('id_in_dataset')}:{row.get('vibe_index')}:{row.get('vibe_scope')}:{row.get('vibe_level')}"


def load_processed_keys(path: Path) -> set[str]:
    """Load keys of already-processed rows for resume."""
    if not path.exists():
        return set()
    processed: set[str] = set()
    for row in iter_jsonl(path):
        processed.add(row_key(row))
    return processed


def score_row(
    row: dict[str, Any],
    scorers: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    """Score a single row with all scorers.

    Args:
        row: Input row with vibe_original and config_payload
        scorers: Dict of scorer name -> scorer function

    Returns:
        Row with added scores dict
    """
    vibe = str(row.get("vibe_original", ""))
    config = row.get("config_payload")

    output_row = dict(row)
    scores: dict[str, dict[str, Any]] = {}

    if config is None:
        # No valid config to score
        for name in scorers:
            scores[name] = {"error": "no_config", "final_score": 0.0}
    else:
        for name, scorer_fn in scorers.items():
            try:
                result = scorer_fn(vibe, config)

                # Validate result implements ScoreResult protocol
                # Wrap dict results in DictScoreResult for protocol validation
                score_result = DictScoreResult(result) if isinstance(result, dict) else result
                validate_score_result(score_result, source=name)

                # Store the raw dict result (with final_score validated)
                scores[name] = result if isinstance(result, dict) else {"final_score": score_result.final_score}
            except Exception as exc:
                _LOGGER.warning("Scorer '%s' failed: %s", name, exc)
                scores[name] = {"error": str(exc), "final_score": 0.0}

    output_row["scores_external"] = scores
    return output_row


def write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    """Write run configuration to output directory."""
    config = {
        "scorers": args.scorers,
        "llm_model": args.llm_model,
        "audio_duration": args.audio_duration,
    }
    config_path = output_dir / RUN_CONFIG_NAME
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    """Main entry point."""
    import os

    from tqdm import tqdm  # type: ignore[import]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Find input files
    split_files: list[tuple[str, Path]] = []
    for split_file in sorted(input_dir.glob("*.jsonl")):
        split_name = split_file.stem
        if args.only_splits and split_name not in args.only_splits:
            continue
        split_files.append((split_name, split_file))

    if not split_files:
        raise SystemExit(f"No JSONL files found in {input_dir}")

    # Check output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, _ in split_files:
        output_path = output_dir / f"{split_name}.jsonl"
        if output_path.exists() and not args.overwrite and not args.resume:
            raise SystemExit(
                f"Output exists: {output_path}. Use --overwrite or --resume."
            )

    # Load environment
    load_env_file(args.env_file)

    # Resolve LLM API key
    llm_api_key = args.llm_api_key or os.environ.get(args.llm_api_key_env)

    # Parse scorers
    scorer_specs = [s.strip() for s in args.scorers.split(",")]
    _LOGGER.info("Parsing scorers: %s", scorer_specs)

    scorers = parse_scorers(
        scorer_specs,
        llm_model=args.llm_model,
        llm_api_key=llm_api_key,
        audio_duration=args.audio_duration,
    )
    _LOGGER.info("Loaded %d scorers: %s", len(scorers), list(scorers.keys()))

    # Write run config
    write_run_config(output_dir, args)

    # Process each split
    total_processed = 0
    for split_name, split_file in split_files:
        output_path = output_dir / f"{split_name}.jsonl"

        # Load processed keys for resume
        processed_keys: set[str] = set()
        if args.resume and output_path.exists():
            processed_keys = load_processed_keys(output_path)
            _LOGGER.info("Resuming %s: %d already processed", split_name, len(processed_keys))

        # Overwrite mode: clear existing file
        if args.overwrite and output_path.exists():
            output_path.unlink()

        # Load input rows
        input_rows = list(iter_jsonl(split_file))
        if args.limit > 0:
            remaining = args.limit - total_processed
            if remaining <= 0:
                break
            input_rows = input_rows[:remaining]

        # Filter out already processed
        pending_rows = [
            row for row in input_rows if row_key(row) not in processed_keys
        ]

        if not pending_rows:
            _LOGGER.info("Split %s: all rows already processed", split_name)
            continue

        _LOGGER.info("Scoring %s: %d pending rows", split_name, len(pending_rows))

        # Setup periodic writer for this split
        writer = SyncPeriodicWriter(
            output_path,
            interval_seconds=args.write_interval,
            overwrite=False,  # Never overwrite in append/resume mode
        )
        writer.start()
        _LOGGER.info(
            "Output file: %s (updates every %.0fs)",
            output_path,
            args.write_interval,
        )

        try:
            with tqdm(total=len(pending_rows), desc=f"Scoring {split_name}") as progress:
                for row in pending_rows:
                    try:
                        result = score_row(row, scorers)

                        # Add to periodic writer buffer
                        writer.add_row(result)

                        progress.update(1)
                        total_processed += 1

                    except Exception as exc:
                        _LOGGER.error("Fatal error on row: %s", exc, exc_info=True)
                        raise
        finally:
            writer.stop()
            _LOGGER.info(
                "Split %s: wrote %d rows total",
                split_name,
                writer.total_written,
            )

        _LOGGER.info("Completed %s: wrote to %s", split_name, output_path)

    _LOGGER.info("Done! Total scored: %d rows", total_processed)


if __name__ == "__main__":
    main()
