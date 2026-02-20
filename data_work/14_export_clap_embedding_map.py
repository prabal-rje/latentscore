"""Export CLAP audio embeddings + config payloads for fast_heavy lookup.

Renders each config as 60s audio, extracts CLAP audio embeddings (512-dim),
and writes a JSONL file suitable for the fast_heavy model.

Run:
    python -m data_work.14_export_clap_embedding_map --limit 5          # test run
    python -m data_work.14_export_clap_embedding_map --workers 4        # parallel
    python -m data_work.14_export_clap_embedding_map --resume           # resume
    python -m data_work.14_export_clap_embedding_map --upload           # upload to HF
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from data_work.lib.jsonl_io import iter_jsonl
from latentscore.audio import write_wav
from latentscore.config import MusicConfig, MusicConfigPrompt
from latentscore.synth import assemble

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_LOCAL = Path("data_work/2026-01-26_scored/vibe_and_embeddings_to_config_map.jsonl")
DEFAULT_INPUT_HF_FILE = "2026-01-26_scored/vibe_and_embeddings_to_config_map.jsonl"
DEFAULT_OUTPUT = Path("data_work/2026-01-26_scored/vibe_and_clap_audio_embeddings_to_config_map.jsonl")
DEFAULT_PROGRESS = Path("data_work/2026-01-26_scored/_clap_embed_progress.jsonl")
DEFAULT_DURATION = 60.0

HF_REPO = "guprab/latentscore-data"
HF_PATH = "2026-01-26_scored/vibe_and_clap_audio_embeddings_to_config_map.jsonl"


def _row_id(row: dict[str, Any]) -> str:
    """Stable identifier for a row."""
    dataset = row.get("dataset", "")
    id_in_dataset = row.get("id_in_dataset", "")
    return f"{dataset}:{id_in_dataset}"


def _load_completed_ids(progress_path: Path) -> set[str]:
    """Load already-completed row IDs from progress file."""
    completed: set[str] = set()
    if not progress_path.exists():
        return completed
    for row in iter_jsonl(progress_path):
        rid = row.get("row_id")
        if isinstance(rid, str):
            completed.add(rid)
    return completed


def _parse_config(config_dict: dict[str, Any]) -> MusicConfig:
    """Parse config dict to MusicConfig, trying prompt labels first."""
    try:
        prompt_config = MusicConfigPrompt.model_validate(config_dict)
        return prompt_config.to_config()
    except Exception:
        return MusicConfig.model_validate(config_dict)


def _render_and_embed_row(
    row: dict[str, Any],
    clap_model: Any,
    duration: float,
) -> dict[str, Any] | None:
    """Render a single config to audio, extract CLAP embedding, return output row."""
    config_dict = row.get("config")
    vibe = row.get("vibe_original")
    if not config_dict or not isinstance(config_dict, dict) or not vibe:
        return None

    try:
        music_config = _parse_config(config_dict)
    except Exception:
        LOGGER.warning("Skipping row %s: config parse failed", _row_id(row))
        return None

    internal = music_config.to_internal()
    audio = assemble(internal, duration=duration)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "sample.wav"
        write_wav(wav_path, audio)
        audio_embed = clap_model.get_audio_embedding_from_filelist([str(wav_path)])

    embed_vec = np.asarray(audio_embed[0], dtype=np.float32)
    norm = float(np.linalg.norm(embed_vec))
    if norm > 0:
        embed_vec = embed_vec / norm

    return {
        "dataset": row.get("dataset"),
        "id_in_dataset": row.get("id_in_dataset"),
        "split": row.get("split"),
        "vibe_original": vibe,
        "clap_audio_embedding": [float(x) for x in embed_vec.tolist()],
        "title": row.get("title"),
        "config": config_dict,
        "palettes": row.get("palettes"),
    }


def _worker_init() -> None:
    """Per-worker initializer: load CLAP model into global."""
    import laion_clap  # type: ignore[import]

    global _WORKER_CLAP  # noqa: PLW0603
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    _WORKER_CLAP = model  # type: ignore[name-defined]


def _worker_process_row(args: tuple[dict[str, Any], float]) -> dict[str, Any] | None:
    """Process a single row in a worker (uses global CLAP model)."""
    row, duration = args
    return _render_and_embed_row(row, _WORKER_CLAP, duration)  # type: ignore[name-defined]


def _process_single(
    rows: list[dict[str, Any]],
    duration: float,
    output_path: Path,
    progress_path: Path,
) -> int:
    """Process rows in single-process mode with streaming writes."""
    import laion_clap  # type: ignore[import]

    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt()
    LOGGER.info("CLAP model loaded")

    written = 0
    with (
        output_path.open("a", encoding="utf-8") as out_handle,
        progress_path.open("a", encoding="utf-8") as prog_handle,
    ):
        for i, row in enumerate(rows):
            t0 = time.monotonic()
            result = _render_and_embed_row(row, clap_model, duration)
            elapsed = time.monotonic() - t0
            if result is not None:
                out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_handle.flush()
                prog_handle.write(json.dumps({"row_id": _row_id(row)}) + "\n")
                prog_handle.flush()
                written += 1
            LOGGER.info(
                "[%d/%d] %s (%.1fs)%s",
                i + 1,
                len(rows),
                _row_id(row),
                elapsed,
                "" if result else " SKIPPED",
            )
    return written


def _process_parallel(
    rows: list[dict[str, Any]],
    duration: float,
    workers: int,
    output_path: Path,
    progress_path: Path,
) -> int:
    """Process rows with multiprocessing pool."""
    args_list = [(row, duration) for row in rows]
    written = 0

    with multiprocessing.Pool(workers, initializer=_worker_init) as pool:
        with (
            output_path.open("a", encoding="utf-8") as out_handle,
            progress_path.open("a", encoding="utf-8") as prog_handle,
        ):
            for i, result in enumerate(pool.imap(_worker_process_row, args_list)):
                row = rows[i]
                if result is not None:
                    out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_handle.flush()
                    prog_handle.write(json.dumps({"row_id": _row_id(row)}) + "\n")
                    prog_handle.flush()
                    written += 1
                if (i + 1) % 50 == 0 or i + 1 == len(rows):
                    LOGGER.info("[%d/%d] written=%d", i + 1, len(rows), written)

    return written


def _upload_to_hf(file_path: Path) -> None:
    """Upload the CLAP embedding map to HuggingFace."""
    from dotenv import load_dotenv  # type: ignore[import]
    from huggingface_hub import HfApi  # type: ignore[import]

    load_dotenv()
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=HF_PATH,
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    LOGGER.info("Uploaded %s to %s/%s", file_path, HF_REPO, HF_PATH)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export CLAP audio embeddings for fast_heavy model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=None, help="Source JSONL (auto-downloads from HF if omitted).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL.")
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS, help="Progress file.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Audio seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0=all).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--resume", action="store_true", help="Resume from progress file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file.")
    parser.add_argument("--upload", action="store_true", help="Upload result to HuggingFace.")
    args = parser.parse_args(argv)

    input_path: Path
    if args.input is not None:
        input_path = args.input.expanduser().resolve()
    elif DEFAULT_INPUT_LOCAL.exists():
        input_path = DEFAULT_INPUT_LOCAL.expanduser().resolve()
    else:
        from huggingface_hub import hf_hub_download  # type: ignore[import]

        input_path = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=DEFAULT_INPUT_HF_FILE,
            )
        )
        LOGGER.info("Downloaded input from HuggingFace: %s", input_path)
    output_path = args.output.expanduser().resolve()
    progress_path = args.progress.expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    # Load all input rows
    all_rows = list(iter_jsonl(input_path))
    LOGGER.info("Loaded %d rows from %s", len(all_rows), input_path)

    # Resume: filter out completed rows
    if args.resume:
        completed = _load_completed_ids(progress_path)
        before = len(all_rows)
        all_rows = [r for r in all_rows if _row_id(r) not in completed]
        LOGGER.info("Resume: %d done, %d remaining", before - len(all_rows), len(all_rows))
    elif not args.overwrite and output_path.exists():
        raise SystemExit(f"Output exists: {output_path} (use --overwrite or --resume)")
    else:
        # Fresh start: truncate output and progress files
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        progress_path.write_text("", encoding="utf-8")

    if args.limit:
        all_rows = all_rows[: args.limit]

    if not all_rows:
        LOGGER.info("Nothing to process.")
        if args.upload and output_path.exists():
            _upload_to_hf(output_path)
        return

    LOGGER.info("Processing %d rows (duration=%.0fs, workers=%d)", len(all_rows), args.duration, args.workers)
    t0 = time.monotonic()

    if args.workers > 1:
        written = _process_parallel(all_rows, args.duration, args.workers, output_path, progress_path)
    else:
        written = _process_single(all_rows, args.duration, output_path, progress_path)

    elapsed = time.monotonic() - t0
    LOGGER.info("Done: %d rows written in %.1fs (%.2fs/row)", written, elapsed, elapsed / max(written, 1))

    if args.upload:
        _upload_to_hf(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
