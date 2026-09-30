from wearusfm.tokenizer_checks.run_io import argv_without_option


def test_argv_without_option_both_forms():
    assert argv_without_option(["--arrays", "a", "--stage", "all", "--only", "b"], "--stage") == ["--arrays", "a", "--only", "b"]
    assert argv_without_option(["--stage=all", "--arrays", "a"], "--stage") == ["--arrays", "a"]
    assert argv_without_option(["--stages", "x", "--arrays", "a"], "--stage") == ["--stages", "x", "--arrays", "a"]  # altra opzione
    assert argv_without_option([], "--stage") == []
