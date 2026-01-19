from __future__ import annotations

from pydantic import BaseModel

from data_work.lib.periodic_writer import SyncPeriodicWriter


class _Row(BaseModel):
    foo: str


def test_periodic_writer_serializes_pydantic(tmp_path) -> None:
    writer = SyncPeriodicWriter(tmp_path / "out.jsonl", interval_seconds=0.0, overwrite=True)
    writer.start()
    writer.add_row(_Row(foo="bar"))
    writer.stop()
    contents = (tmp_path / "out.jsonl").read_text(encoding="utf-8").strip()
    assert '"foo": "bar"' in contents
