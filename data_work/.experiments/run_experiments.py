#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_commands(md_path: Path) -> list[str]:
    commands: list[str] = []
    in_block = False
    block_lines: list[str] = []

    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```bash"):
            in_block = True
            block_lines = []
            continue
        if in_block and line.startswith("```"):
            in_block = False
            commands.extend(_block_to_commands(block_lines))
            block_lines = []
            continue
        if in_block:
            block_lines.append(line)

    return commands


def _block_to_commands(lines: list[str]) -> list[str]:
    commands: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            commands.append(buffer.strip())
        buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            flush()
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1].rstrip() + " "
        else:
            buffer += stripped
            flush()

    flush()
    return commands


def run_commands(commands: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for index, cmd in enumerate(commands, start=1):
            log.write(f"=== COMMAND {index} ===\n{cmd}\n")
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            log.write(f"exit={proc.returncode}\n")
            if proc.stdout:
                log.write("stdout:\n")
                log.write(proc.stdout)
                if not proc.stdout.endswith("\n"):
                    log.write("\n")
            if proc.stderr:
                log.write("stderr:\n")
                log.write(proc.stderr)
                if not proc.stderr.endswith("\n"):
                    log.write("\n")
            log.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EXPERIMENTS.md commands and log output.")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional substring filter to select commands to run.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("data_work/.experiments/run_logs/run_experiments.log"),
        help="Log file path.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data_work/EXPERIMENTS.md"),
        help="Path to EXPERIMENTS.md.",
    )
    args = parser.parse_args()

    all_commands = parse_commands(args.file)
    if args.filter:
        commands = [cmd for cmd in all_commands if args.filter in cmd]
    else:
        commands = all_commands

    run_commands(commands, args.log)
