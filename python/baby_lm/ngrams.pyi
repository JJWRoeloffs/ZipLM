import enum
from typing import List

from baby_lm.utils import PyCorpusSentences

class PyNGramModel:
    def __init__(self, data: PyCorpusTokens, n: int) -> None: ...
    def sanitize(self, data: List[PyToken]) -> List[PyToken]: ...
    def get_log_likelyhood(self, data: List[PyToken]) -> float: ...

class PyTokenKind(enum.Enum):
    Lemma = 0
    Boundry = 1
    Unknown = 2

class PyToken:
    kind: PyTokenKind
    data: str

class PyCorpusTokens:
    def __init__(self, items: List[List[PyToken]]) -> None: ...

def tokenkind_from_str(string: str) -> PyTokenKind: ...
def tokenize_pycorpus(sentences: PyCorpusSentences) -> PyCorpusTokens: ...
def pytokenize(sentence: str) -> List[PyToken]: ...