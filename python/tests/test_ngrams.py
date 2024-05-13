import pytest
import baby_lm


def test_tokenkind():
    assert baby_lm.ngrams.PyTokenKind(0) == baby_lm.ngrams.tokenkind_from_str("lemma")


def test_pytokens():
    tokenkind = baby_lm.ngrams.tokenkind_from_str("lemma")
    token = baby_lm.ngrams.PyToken(tokenkind, "hello")
    assert token.data == "hello"

    with pytest.raises(ValueError):
        baby_lm.ngrams.PyToken(baby_lm.ngrams.tokenkind_from_str("unknown"), "test")

    with pytest.raises(ValueError):
        baby_lm.ngrams.PyToken(baby_lm.ngrams.tokenkind_from_str("boundry"), "test")


def test_pytokenize():
    tokens = baby_lm.ngrams.pytokenize("This is a test sentence")
    assert all(isinstance(token, baby_lm.ngrams.PyToken) for token in tokens)


class TestWithCorpus:
    items = [
        "This is a test sentence.",
        "This is another test sentence.",
        "This is a third test sentence.",
    ]
    sentences = baby_lm.utils.PyCorpusSentences(items)

    def test_pytoken_generation(self):
        corpus = baby_lm.ngrams.tokenize_pycorpus(self.sentences)
        assert all(
            isinstance(token, baby_lm.ngrams.PyToken)
            for sentence in corpus.items()
            for token in sentence
        )

    def test_ngram_sanitize(self):
        corpus = baby_lm.ngrams.tokenize_pycorpus(self.sentences)
        model = baby_lm.ngrams.PyNGramModel(corpus, 3)
        tokens = model.sanitize(baby_lm.ngrams.pytokenize("This unknown"))

        assert tokens[0].data == "This"
        assert tokens[0].kind == baby_lm.ngrams.tokenkind_from_str("lemma")
        assert tokens[1].data == ""
        assert tokens[1].kind == baby_lm.ngrams.tokenkind_from_str("unknown")

    def test_log_likelyhood(self):
        corpus = baby_lm.ngrams.tokenize_pycorpus(self.sentences)
        model = baby_lm.ngrams.PyNGramModel(corpus, 3)
        assert 0 == model.get_log_likelyhood(
            model.sanitize(baby_lm.ngrams.pytokenize("This unkown"))
        )

        likelyhood = model.get_log_likelyhood(
            model.sanitize(baby_lm.ngrams.pytokenize("This is a test sentence"))
        )
        assert 0 < (10**likelyhood) < 1
