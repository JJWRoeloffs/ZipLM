import sys
import json
import time
import random
import argparse
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from zip_lm.eval import evaluate, get_testdata_subset
from zip_lm.readdata import BlimpPyItem, get_blimp_data


from typing import List, Literal


@dataclass
class Args:
    seed: int
    nr_per_type: int
    model_name: Literal["t5-small", "t5-base", "t5-large"]


def parse_args(args: List[str]) -> Args:
    parser = argparse.ArgumentParser(
        prog="evaluate_t5",
        description="The script to use to evaluate a pretrained t5 model",
    )
    parser.add_argument("seed", type=int, help="The seed to use for the rng")
    parser.add_argument(
        "nr_per_type", type=int, help="The amount of blimp items to get per type"
    )
    parser.add_argument(
        "--model_name",
        choices=["t5-small", "t5-base", "t5-large"],
        default="t5-small",
        nargs="?",
        help="The model to run",
    )
    arguments = parser.parse_args(args)
    return Args(arguments.seed, arguments.nr_per_type, arguments.model_name)


def calculate_log_likelihoods(
    items: List[BlimpPyItem], model: T5ForConditionalGeneration, tokenizer: T5Tokenizer
):
    def compute_ll(sentence: str) -> float:
        inputs = tokenizer(sentence, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        labels = input_ids.clone()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return -outputs.loss.item() * (labels.size(1) - 1)

    model.eval()

    with torch.no_grad():
        for item in items:
            item.ll_sentence_bad = compute_ll(item.sentence_bad)
            item.ll_sentence_good = compute_ll(item.sentence_good)


def run(args: Args) -> None:
    before = time.time()
    random.seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        args.model_name
    )  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # type: ignore
    print(f"Retrieved {args.model_name} in {time.time() - before:.2f} seconds")

    before = time.time()
    testdata = get_blimp_data("blimp/data")
    inputdata = get_testdata_subset(testdata, args.nr_per_type)
    print(f"Retrieved testing data in {time.time() - before:.2f} seconds")

    before = time.time()
    calculate_log_likelihoods(inputdata, model, tokenizer)
    print(f"Calculated lls with {args.model_name} in {time.time()-before:.2f} seconds")

    results = {
        "seed": args.seed,
        "type": args.model_name,
        "evaluation_subset": args.nr_per_type,
        "results": evaluate(inputdata),
    }
    pprint(results)

    results_dir = Path() / "results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "t5model.jsonl"
    with results_file.open("a", encoding="utf-8") as f:
        json.dump(results, f)
        f.write("\n")


def main(args: List[str]) -> None:
    run(parse_args(args))


if __name__ == "__main__":
    main(sys.argv[1:])
