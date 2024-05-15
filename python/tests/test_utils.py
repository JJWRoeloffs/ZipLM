from zip_lm.utils import PyCorpusSentences


def test_pycorpussents():
    items = [
        "This is a test sentence.",
        "This is another test sentence.",
        "This is a third test sentence.",
    ]
    corpus = PyCorpusSentences(items)
    assert corpus.items() == items
