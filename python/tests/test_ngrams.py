import pytest
from zip_lm.ngrams import (
    PyTokenKind,
    PyToken,
    pytokenize,
    tokenize_pycorpus,
    tokenkind_from_str,
    PyNGramModel,
)
from zip_lm.utils import PyCorpusSentences


def test_tokenkind():
    assert PyTokenKind(0) == tokenkind_from_str("lemma")


def test_pytokens():
    token = PyToken(tokenkind_from_str("lemma"), "hello")
    assert token.data == "hello"

    with pytest.raises(ValueError):
        PyToken(tokenkind_from_str("unknown"), "test")

    with pytest.raises(ValueError):
        PyToken(tokenkind_from_str("boundry"), "test")


def test_pytokenize():
    tokens = pytokenize("This is a test sentence")
    assert all(isinstance(token, PyToken) for token in tokens)


class TestWithCorpus:
    items = [
        "This is a test sentence.",
        "This is another test sentence.",
        "This is a third test sentence.",
    ]
    sentences = PyCorpusSentences(items)

    def test_pytoken_generation(self):
        corpus = tokenize_pycorpus(self.sentences)
        assert all(
            isinstance(token, PyToken)
            for sentence in corpus.items()
            for token in sentence
        )

    def test_ngram_sanitize(self):
        model = PyNGramModel(tokenize_pycorpus(self.sentences), 3)
        tokens = model.sanitize(pytokenize("This unknown"))

        assert tokens[0].data == "this"
        assert tokens[0].kind == tokenkind_from_str("lemma")
        assert tokens[1].data == ""
        assert tokens[1].kind == tokenkind_from_str("unknown")

    def test_log_likelyhood(self):
        model = PyNGramModel(tokenize_pycorpus(self.sentences), 3)
        assert 0 == model.get_log_likelyhood(model.sanitize(pytokenize("This unkown")))

        likelyhood = model.get_log_likelyhood(
            model.sanitize(pytokenize("This is a test sentence"))
        )
        assert 0 < (10**likelyhood) < 1
