import importlib

import pytest

MODULE = importlib.import_module("data_work.07_human_eval_pack")


def test_generate_requires_args() -> None:
    with pytest.raises(SystemExit):
        MODULE.main(["generate"])  # type: ignore[arg-type]
