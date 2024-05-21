import sys
import time
import json
import random
import argparse
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass


from zip_lm.eval import evaluate
from zip_lm.eval.get_data import get_testdata_subset
from zip_lm.readdata import get_blimp_data
from zip_lm.eval_utils import get_random_lls

from typing import List


@dataclass
class Args:
    seed: int
    nr_per_type: int


def parse_args(args: List[str]) -> Args:
    parser = argparse.ArgumentParser(
        prog="evaluate_random",
        description="The script that does some benchmark valuation on random lls",
    )
    parser.add_argument("seed", help="The seed to use for the rng")
    parser.add_argument("nr_per_type", help="The amount of blimp items to get per type")
    arguments = parser.parse_args(args)
    return Args(
        int(arguments.seed),
        int(arguments.nr_per_type),
    )


def run(args: Args) -> None:
    random.seed(args.seed)

    before = time.time()
    testdata = get_blimp_data("blimp/data")
    print(f"Retrieved testing data in {time.time() - before:.2f} seconds")

    inputdata = get_testdata_subset(testdata, args.nr_per_type)
    print(f"Testing on {len(inputdata)} Items")

    before = time.time()
    lls = get_random_lls(inputdata, random.randint(0, 2147483647))
    print(f"Calculated lls with ngrams model in {time.time() - before:.2f} seconds")

    results = {
        "seed": args.seed,
        "type": "RandomModel",
        "evaluation_subset": args.nr_per_type,
        "results": evaluate(lls),
    }
    pprint(results)

    results_dir = Path() / "results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "random.jsonl"
    with results_file.open("a", encoding="utf-8") as f:
        json.dump(results, f)
        f.write("\n")


def main(args: List[str]) -> None:
    run(parse_args(args))


if __name__ == "__main__":
    main(sys.argv[1:])
