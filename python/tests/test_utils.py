import baby_lm


def test_pycorpussents():
    items = [
        "This is a test sentence.",
        "This is another test sentence.",
        "This is a third test sentence.",
    ]
    corpus = baby_lm.utils.PyCorpusSentences(items)
    assert corpus.items() == items
