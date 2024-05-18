import random
from pathlib import Path
from collections import defaultdict

from zip_lm.readdata import BlimpPyItem, get_sentence_corpus
from zip_lm.utils import PyCorpusSentences

from typing import Literal, List, Dict, Tuple


def get_training_data(which: Literal["10M", "100M"] = "10M") -> PyCorpusSentences:
    corpus_dir = Path() / "data" / f"train_{which}"
    return get_sentence_corpus(str(corpus_dir))


# As these models take ages to evaulate, we get only a subset of the test data.
def get_testdata_subset(testdata: List[BlimpPyItem], nr_per: int) -> List[BlimpPyItem]:
    newdata: List[BlimpPyItem] = []
    counts: Dict[Tuple[str, str], int] = defaultdict(lambda: 0)
    for item in testdata:
        if counts[(item.field, item.linguistics_term)] < nr_per:
            counts[(item.field, item.linguistics_term)] += 1
            newdata.append(item)
    random.shuffle(newdata)
    return newdata
