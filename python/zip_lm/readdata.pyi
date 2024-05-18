from typing import List
from zip_lm.utils import PyCorpusSentences

class BlimpPyItem:
    sentence_good: str
    ll_sentence_good: float
    sentence_bad: str
    ll_sentence_bad: float
    field: str
    linguistics_term: str
    uid: str
    pair_id: int
    def __init__(
        self,
        sentence_good: str,
        ll_sentence_good: float,
        sentence_bad: str,
        ll_sentence_bad: float,
        field: str,
        linguistics_term: str,
        uid: str,
        pair_id: int,
    ): ...

def get_blimp_data(path: str) -> List[BlimpPyItem]: ...
def get_sentence_corpus(path: str) -> PyCorpusSentences: ...
