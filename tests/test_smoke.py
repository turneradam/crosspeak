import crosspeak


def test_version():
    import re
    assert re.fullmatch(r"\d+\.\d+\.\d+", crosspeak.__version__)
