"""Merge LoRA adapters and export models to GGUF/MLC formats."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict

LOGGER = logging.getLogger(__name__)

DEFAULT_GGUF_QUANT = "q4_k_m"
DEFAULT_MLC_QUANT = "q4f16_1"
DEFAULT_MERGED_NAME = "data_work/.exports/combined-model"


class ExportConfig(BaseModel):
    """Configuration for merge + export workflows."""

    model_config = ConfigDict(extra="forbid")

    base_model: str | None
    adapter_path: Path | None
    merged_output: Path
    overwrite: bool
    gguf: bool
    gguf_quantize: str
    gguf_command: str
    mlc: bool
    mlc_quantize: str
    mlc_command: str


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output path already exists: {path}. Use --overwrite to replace.")
    path.mkdir(parents=True, exist_ok=True)


def _run_command(command: str, cwd: Path | None = None) -> None:
    LOGGER.info("Running command: %s", command)
    subprocess.run(
        shlex.split(command),
        check=True,
        cwd=str(cwd) if cwd else None,
    )


def _merge_adapter(config: ExportConfig) -> None:
    if config.base_model is None or config.adapter_path is None:
        raise SystemExit("--base-model and --adapter are required to merge adapters.")
    _ensure_output_dir(config.merged_output, config.overwrite)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForCausalLM.from_pretrained(config.base_model)
    model = PeftModel.from_pretrained(model, str(config.adapter_path))
    merged = model.merge_and_unload()
    merged.save_pretrained(config.merged_output)
    tokenizer.save_pretrained(config.merged_output)


def _export_gguf(config: ExportConfig) -> None:
    output_path = config.merged_output
    command = config.gguf_command.format(
        model_path=output_path,
        quantize=config.gguf_quantize,
    )
    _run_command(command)


def _export_mlc(config: ExportConfig) -> None:
    output_path = config.merged_output
    command = config.mlc_command.format(
        model_path=output_path,
        quantize=config.mlc_quantize,
    )
    _run_command(command)


def _build_parser(show_advanced: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge LoRA adapters into full models and export to GGUF/MLC formats. "
            "Use --advanced to reveal extra knobs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Show advanced export options in -h output.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base HF model to merge adapters into (optional if no merge).",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Path to LoRA adapter directory to merge.",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=Path(DEFAULT_MERGED_NAME),
        help="Output directory for merged full-precision model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite merged-output directory if it exists.",
    )
    parser.add_argument(
        "--gguf",
        action="store_true",
        help="Export GGUF format (llama.cpp).",
    )
    parser.add_argument(
        "--gguf-quantize",
        type=str,
        default=DEFAULT_GGUF_QUANT,
        help="GGUF quantization type.",
    )
    parser.add_argument(
        "--mlc",
        action="store_true",
        help="Export MLC format (WebGPU).",
    )
    parser.add_argument(
        "--mlc-quantize",
        type=str,
        default=DEFAULT_MLC_QUANT,
        help="MLC quantization type.",
    )
    parser.add_argument(
        "--gguf-command",
        type=str,
        default="python -m src.export.to_gguf {model_path} --quantize {quantize}",
        help=(
            "Advanced: custom GGUF export command template. Placeholders: {model_path}, {quantize}."
        )
        if show_advanced
        else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mlc-command",
        type=str,
        default="python -m src.export.to_mlc {model_path} --quantize {quantize}",
        help=(
            "Advanced: custom MLC export command template. Placeholders: {model_path}, {quantize}."
        )
        if show_advanced
        else argparse.SUPPRESS,
    )

    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--advanced", action="store_true")
    known, _ = pre_parser.parse_known_args(argv)
    parser = _build_parser(show_advanced=known.advanced)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv if argv is not None else os.sys.argv[1:])

    config = ExportConfig(
        base_model=args.base_model,
        adapter_path=args.adapter,
        merged_output=args.merged_output,
        overwrite=args.overwrite,
        gguf=args.gguf,
        gguf_quantize=args.gguf_quantize,
        gguf_command=getattr(
            args,
            "gguf_command",
            "python -m src.export.to_gguf {model_path} --quantize {quantize}",
        ),
        mlc=args.mlc,
        mlc_quantize=args.mlc_quantize,
        mlc_command=getattr(
            args,
            "mlc_command",
            "python -m src.export.to_mlc {model_path} --quantize {quantize}",
        ),
    )

    if config.adapter_path is not None:
        _merge_adapter(config)

    if config.gguf:
        _export_gguf(config)

    if config.mlc:
        _export_mlc(config)

    LOGGER.info("Export workflow completed.")


if __name__ == "__main__":
    main()
