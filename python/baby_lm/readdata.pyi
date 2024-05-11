from typing import List
from baby_lm.utils import PyCorpusSentences

class BlimpPyItem:
    sentence_good: str
    sentence_bad: str
    field: str
    linguistics_term: str
    uid: str
    pair_id: int
    def __init__(
        self,
        sentence_good: str,
        sentence_bad: str,
        field: str,
        linguistics_term: str,
        uid: str,
        pair_id: int,
    ): ...

def get_blimp_data(path: str) -> List[BlimpPyItem]: ...
def get_sentence_corpus(path: str) -> PyCorpusSentences: ...
