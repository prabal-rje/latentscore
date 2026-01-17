import importlib


def test_parse_model_kwargs_accepts_mapping() -> None:
    module = importlib.import_module("data_work.02_process_base_data")
    parse = getattr(module, "_parse_model_kwargs")
    assert parse({"foo": "bar"}, label="--model-kwargs") == {"foo": "bar"}
