from sample import core, helpers


def test_basic_imports():
    assert callable(core.run_analysis)
    assert callable(helpers.extract_airlines)
