from data_work.vibe_schema import split_records


def test_split_records_deterministic() -> None:
    records = [f"item-{index}" for index in range(10)]
    first = split_records(records, seed=123)
    second = split_records(records, seed=123)
    assert first == second
    assert sum(len(items) for items in first.values()) == len(records)
