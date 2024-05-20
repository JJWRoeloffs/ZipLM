import sys
import time
import json
import argparse
import random
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass


from zip_lm.eval import get_training_data, evaluate, get_testdata_subset
from zip_lm.zipmodels import PySoftmaxZipModel, bytes_pycorpus
from zip_lm.readdata import get_blimp_data
from zip_lm.eval_utils import get_softmax_lls

from typing import List, Literal


T_TYPES = Literal["mean", "quarterq", "halfq", "threequarterq"]
T_CONVERSION = {"mean": 0, "quarterq": 1, "halfq": 2, "threequarterq": 3}


@dataclass
class Args:
    seed: int
    nr_per_type: int
    sample_size: int
    t_type: T_TYPES
    data: Literal["10M", "100M"]


def parse_args(args: List[str]) -> Args:
    parser = argparse.ArgumentParser(
        prog="evaluate_softmaxzipmodel",
        description="The script that does the evaluation of the softmax model",
    )
    parser.add_argument("seed", help="The seed to use for the rng")
    parser.add_argument("nr_per_type", help="The amount of blimp items to get per type")
    parser.add_argument("sample_size", help="The sample size used in the softmax model")
    parser.add_argument(
        "-t",
        help="The agrevation method used in the softmax model",
        choices=T_CONVERSION.keys(),
        default="halfq",
        nargs="?",
        required=False,
    )
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
        int(arguments.sample_size),
        arguments.t,
        arguments.data,
    )


def run(args: Args) -> None:
    random.seed(args.seed)
    before = time.time()
    model = PySoftmaxZipModel(
        bytes_pycorpus(get_training_data(args.data)),
        args.sample_size,
        T_CONVERSION[args.t_type],
        random.randint(0, 2147483647),
    )
    print(f"Trained SZ model with sample_size={args.sample_size} with t={args.t_type}")
    print(f"on the {args.data} training set in {time.time() - before:.2f} seconds")

    before = time.time()
    testdata = get_blimp_data("blimp/data")
    print(f"Retrieved testing data in {time.time() - before:.2f} seconds")

    inputdata = get_testdata_subset(testdata, args.nr_per_type)
    print(f"Testing on {len(inputdata)} Items")

    before = time.time()
    lls = get_softmax_lls(model, inputdata)
    print(f"Calculated lls with softmax model in {time.time() - before:.2f} seconds")

    results = {
        "seed": args.seed,
        "type": "PySoftmaxZipModel",
        "params": {
            "sample_size": args.sample_size,
            "t_type": args.t_type,
        },
        "training_data": args.data,
        "evaluation_subset": args.nr_per_type,
        "results": evaluate(lls),
    }
    pprint(results)

    results_dir = Path() / "results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "softmaxzip.jsonl"
    with results_file.open("a", encoding="utf-8") as f:
        json.dump(results, f)
        f.write("\n")


def main(args: List[str]) -> None:
    run(parse_args(args))


if __name__ == "__main__":
    main(sys.argv[1:])
