from zip_lm.utils import PyCorpusSentences
from zip_lm.zipmodels import (
    PyBootstrapZipModel,
    PySoftmaxZipModel,
    bytes_pycorpus,
    PyCorpusBytes,
)


class TestWithCorpus:
    items = [
        "This is a test sentence.",
        "This is another test sentence.",
        "This is a third test sentence.",
    ]
    sentences = PyCorpusSentences(items)

    def test_corpus_bytes(self):
        corpus = PyCorpusBytes([string.encode("utf-8") for string in self.items])
        byte_arrays = [bytes(item) for item in corpus.items()]
        assert all(isinstance(item.decode(), str) for item in byte_arrays)

    def test_bytes_pycorpus(self):
        corpus = bytes_pycorpus(self.sentences)
        byte_arrays = [bytes(item) for item in corpus.items()]
        assert all(isinstance(item.decode(), str) for item in byte_arrays)

    def test_pybootstrapmodel(self):
        corpus = bytes_pycorpus(self.sentences)
        model = PyBootstrapZipModel(corpus, 2, 2, 0)
        likelyhood = model.get_log_likelyhood("This is a test sentence".encode("utf-8"))
        assert 0 < (10**likelyhood) < 1

    def test_pysoftmaxmodel(self):
        corpus = bytes_pycorpus(self.sentences)
        model = PySoftmaxZipModel(corpus, 2, 0)
        likelyhood = model.get_log_likelyhood("This is a test".encode("utf-8"))
        assert 0 < (10**likelyhood) < 1
