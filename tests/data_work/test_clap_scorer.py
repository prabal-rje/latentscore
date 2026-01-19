def test_clap_scorer_class_exposed() -> None:
    import data_work.lib.clap_scorer as clap_scorer

    assert hasattr(clap_scorer, "ClapScorer")
