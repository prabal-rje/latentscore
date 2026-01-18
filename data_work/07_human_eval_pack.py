"""Human eval pack placeholder (not implemented yet)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and analyze human eval packs (not implemented yet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a human eval pack")
    generate.add_argument("--model-a")
    generate.add_argument("--model-b")
    generate.add_argument("--eval-set")
    generate.add_argument("--n-samples", type=int)
    generate.add_argument("--output-dir")

    analyze = subparsers.add_parser("analyze", help="Analyze human eval results")
    analyze.add_argument("--input-dir")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    parser.parse_args(argv)
    raise NotImplementedError("07_human_eval_pack is not implemented yet.")


if __name__ == "__main__":
    main()
