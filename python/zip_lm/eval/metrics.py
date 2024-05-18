from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score

from zip_lm.readdata import BlimpPyItem

from typing import Dict, List


def evaluate(items: List[BlimpPyItem]) -> Dict[str, float]:
    golden = [True for _ in items]
    results = [item.ll_sentence_bad < item.ll_sentence_good for item in items]

    # The official BabyLM pipeline also uses stuff like Bleu, ChrF++, and more like that.
    # But, with how simple this system is, I don't think that is meaningful.
    return {
        "f1": f1_score(golden, results),
        "matthews": matthews_corrcoef(golden, results),
        "accuracy": accuracy_score(golden, results),
        "recall": recall_score(golden, results),
        "all_accuracy": all_accuracy(items),
    }


# Accuracy that is only true if _all_ of a given type of question are correct
def all_accuracy(items: List[BlimpPyItem]) -> float:
    types = {(i.field, i.linguistics_term) for i in items}
    groups = [[g for g in items if (g.field, g.linguistics_term) == i] for i in types]
    scores = [all(i.ll_sentence_bad < i.ll_sentence_good for i in g) for g in groups]
    return accuracy_score([True for _ in scores], scores)
