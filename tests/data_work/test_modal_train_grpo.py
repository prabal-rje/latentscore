import importlib


def test_ensure_lora_trainable_sets_requires_grad() -> None:
    modal_train = importlib.import_module("data_work.03_modal_train")

    class DummyParam:
        def __init__(self) -> None:
            self.requires_grad = False
            self._numel = 3

        def numel(self) -> int:
            return self._numel

    class DummyModel:
        def __init__(self) -> None:
            self._params = {
                "base.weight": DummyParam(),
                "lora_A.weight": DummyParam(),
            }

        def named_parameters(self):
            return list(self._params.items())

    model = DummyModel()
    modal_train._ensure_lora_trainable(model)
    assert model._params["lora_A.weight"].requires_grad is True
    assert model._params["base.weight"].requires_grad is False
