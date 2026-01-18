import importlib

MODULE = importlib.import_module("data_work.03_modal_train")


def test_modal_train_image_includes_soundfile() -> None:
    assert any("soundfile" in pkg for pkg in MODULE.TRAIN_IMAGE_PACKAGES)
    assert any("scipy" in pkg for pkg in MODULE.TRAIN_IMAGE_PACKAGES)
