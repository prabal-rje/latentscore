import importlib

import pytest

MODULE = importlib.import_module("data_work.07_human_eval_pack")


def test_generate_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        MODULE.main(["generate"])  # type: ignore[arg-type]
