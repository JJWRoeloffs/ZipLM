from .readdata import BlimpPyItem
from .ngrams import PyNGramModel
from .zipmodels import PyBootstrapZipModel, PySoftmaxZipModel

from typing import List

def get_ngrams_lls(
    model: PyNGramModel, data: List[BlimpPyItem]
) -> List[BlimpPyItem]: ...
def get_bootstrap_lls(
    model: PyBootstrapZipModel, data: List[BlimpPyItem]
) -> List[BlimpPyItem]: ...
def get_softmax_lls(
    model: PySoftmaxZipModel, data: List[BlimpPyItem]
) -> List[BlimpPyItem]: ...
def get_random_lls(data: List[BlimpPyItem], seed: int) -> List[BlimpPyItem]: ...
