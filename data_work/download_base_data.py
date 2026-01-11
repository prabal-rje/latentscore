"""Module entrypoint for 01_download_base_data.py."""

from __future__ import annotations

from data_work._script_runner import run_script


def main() -> None:
    run_script("01_download_base_data.py")


if __name__ == "__main__":
    main()
