import sys
import time
import json
import random
import argparse
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass


from zip_lm.eval import get_training_data, evaluate
from zip_lm.eval.get_data import get_testdata_subset
from zip_lm.ngrams import tokenize_pycorpus, PyNGramModel
from zip_lm.readdata import get_blimp_data
from zip_lm.eval_utils import get_ngrams_lls

from typing import List, Literal


@dataclass
class Args:
    seed: int
    nr_per_type: int
    n: int
    data: Literal["10M", "100M"]


def parse_args(args: List[str]) -> Args:
    parser = argparse.ArgumentParser(
        prog="evaluate_ngrams",
        description="The script that does the evaluation of the ngrams model",
    )
    parser.add_argument("seed", help="The seed to use for the rng")
    parser.add_argument("nr_per_type", help="The amount of blimp items to get per type")
    parser.add_argument("n", help="The n of the n-gram model")
    parser.add_argument(
        "-d",
        "--data",
        help="The data to use to train on",
        choices=["10M", "100M"],
        default="10M",
        nargs="?",
        required=False,
    )
    arguments = parser.parse_args(args)
    return Args(
        int(arguments.seed),
        int(arguments.nr_per_type),
        int(arguments.n),
        arguments.data,
    )


def run(args: Args) -> None:
    random.seed(args.seed)
    before = time.time()
    model = PyNGramModel(tokenize_pycorpus(get_training_data(args.data)), args.n)
    print(f"Trained NGram model with n={args.n}, on the {args.data} training set.")
    print(f"in {time.time() - before:.2f} seconds")

    before = time.time()
    testdata = get_blimp_data("blimp/data")
    print(f"Retrieved testing data in {time.time() - before:.2f} seconds")

    inputdata = get_testdata_subset(testdata, args.nr_per_type)
    print(f"Testing on {len(inputdata)} Items")

    before = time.time()
    ngram_lls = get_ngrams_lls(model, inputdata)
    print(f"Calculated lls with ngrams model in {time.time() - before:.2f} seconds")

    results = {
        "type": "PyNGramModel",
        "params": {
            "n": args.n,
        },
        "training_data": args.data,
        "evaluation_subset": args.nr_per_type,
        "results": evaluate(ngram_lls),
    }
    pprint(results)

    results_dir = Path() / "results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "ngrams.jsonl"
    with results_file.open("a", encoding="utf-8") as f:
        json.dump(results, f)
        f.write("\n")


def main(args: List[str]) -> None:
    run(parse_args(args))


if __name__ == "__main__":
    main(sys.argv[1:])
