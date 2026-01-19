from pathlib import Path

from pydantic import BaseModel

from data_work.lib.periodic_writer import SyncPeriodicWriter


class DemoRow(BaseModel):
    foo: str
    bar: int
    output_path: Path


def test_sync_writer_serializes_pydantic(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    writer = SyncPeriodicWriter(path, interval_seconds=0.01, overwrite=True)
    writer.start()
    writer.add_row(DemoRow(foo="hi", bar=1, output_path=tmp_path / "demo"))
    writer.stop()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    assert '"foo"' in lines[0]
    assert '"output_path"' in lines[0]
