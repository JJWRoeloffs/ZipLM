import sys
import json
import argparse
from pathlib import Path
from pprint import pprint
from statistics import mean
from itertools import permutations
from collections import Counter
from dataclasses import dataclass

import nltk
from pandas import DataFrame

from zip_lm.readdata import PyDataItems, get_data_items
from zip_lm.ngrams import pytokenize

from typing import Any, Dict, List, Literal, Tuple


@dataclass
class Args:
    data: Literal["10M", "100M"]
    latex: bool
    most_common: int


def parse_args(args: List[str]) -> Args:
    parser = argparse.ArgumentParser(
        prog="get_data_metrics",
        description="The script that returns out the data metrics",
    )
    parser.add_argument(
        "-d",
        "--data",
        help="The data track to get the metrics on",
        choices=["10M", "100M"],
        default="10M",
        nargs="?",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--most_common",
        help="How many of the biggest differences to show",
        type=int,
        default=10,
        nargs="?",
        required=False,
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="If passed, a latex table is saved with the results",
        default=False,
        required=False,
    )
    arguments = parser.parse_args(args)
    return Args(arguments.data, arguments.latex, arguments.most_common)


def get_basic_info_individual(corpus: List[str]) -> Dict[str, Any]:
    words_naive = [t for i in corpus for t in pytokenize(i)]
    sentences_text = [s for i in corpus for s in nltk.sent_tokenize(i)]
    # Yes, you have to pass .lower(): https://www.nltk.org/api/nltk.probability.FreqDist.html
    tokens = [t.lower() for s in sentences_text for t in nltk.word_tokenize(s)]
    tagged = nltk.pos_tag(tokens)

    return {
        # Naive
        "nr_sentences_naive": len(corpus),
        "nr_words": len(words_naive),
        "nr_unique_words": len(set(words_naive)),
        "mean_word_length": mean(len(w.data) for w in words_naive),
        "mean_words_per_sentence": len(words_naive) / len(corpus),
        # NLTK
        "nr_sentences_nltk": len(sentences_text),
        "nr_tokens": len(tokens),
        "nr_unique_tokens": len(set(tokens)),
        "mean_tokens_per_sentence": len(tokens) / len(sentences_text),
        "common_tokens": Counter(t[1] for t in tagged).most_common(10),
    }


def get_basic_info(data: PyDataItems) -> Dict[str, Any]:
    all_data = (
        data.simple_wiki
        + data.childes
        + data.gutenberg
        + data.subtitiles
        + data.simple_wiki
        + data.switchboard
    )
    return {
        "bnc_spoken": get_basic_info_individual(data.bnc_spoken),
        "childes": get_basic_info_individual(data.childes),
        "gutenberg": get_basic_info_individual(data.gutenberg),
        "subtitiles": get_basic_info_individual(data.subtitiles),
        "simple_wiki": get_basic_info_individual(data.simple_wiki),
        "switchboard": get_basic_info_individual(data.switchboard),
        "combined": get_basic_info_individual(all_data),
    }


def basic_info_to_latex(data_dict: Dict[str, Dict[str, Any]], to: Path) -> None:
    # Remove the counts from the most_common_tokens as they clutter up the table
    data_dict = {
        key: {k: v for k, v in val.items() if k != "common_tokens"}
        for key, val in data_dict.items()
    }
    df = DataFrame.from_dict(data_dict).reset_index()
    df.to_latex(to, float_format="%.2f", index=False)


def compare_frequencies(
    freq1: nltk.FreqDist, freq2: nltk.FreqDist, top_n: int = 10
) -> List[Tuple[str, str]]:
    comparisonts = {
        word: freq1[word] / freq1.N() - freq2[word] / freq2.N()
        for word in set(freq1.keys()).union(set(freq2.keys()))
    }
    # We don't use an abs() here, because we have both A-B and B-A
    res = sorted(comparisonts.items(), key=lambda i: i[1], reverse=True)[:top_n]
    return [(i, f"{v:.3f}") for i, v in res]


def get_frequency_info(data: PyDataItems, top_n: int = 10) -> Dict[str, Any]:
    nltk.download("punkt")
    nltk.download("brown")

    def token_freqdist(items: List[str]) -> nltk.FreqDist:
        return nltk.FreqDist(t.lower() for i in items for t in nltk.word_tokenize(i))

    items = {
        "brown": token_freqdist(nltk.corpus.brown.words()),
        "bnc_spoken": token_freqdist(data.bnc_spoken),
        "childes": token_freqdist(data.childes),
        "gutenberg": token_freqdist(data.gutenberg),
        "subtitiles": token_freqdist(data.subtitiles),
        "simple_wiki": token_freqdist(data.simple_wiki),
        "switchboard": token_freqdist(data.switchboard),
    }
    return {
        f"{kv1[0]}-{kv2[0]}": compare_frequencies(kv1[1], kv2[1], top_n)
        for kv1, kv2 in permutations(items.items(), r=2)
    }


def run(args: Args):
    data_dir = Path() / "data" / f"train_{args.data}"
    data = get_data_items(str(data_dir))
    frequency_info = get_frequency_info(data, args.most_common)
    data_dict = get_basic_info(data)

    pprint(data_dict)
    pprint(frequency_info)

    results_dir = Path() / "results"
    results_dir.mkdir(exist_ok=True)

    if args.latex:
        basic_info_to_latex(data_dict, results_dir / "basic_data.tex")

    results = {"basic_info": data_dict, "frequency_info": frequency_info}

    results_file = results_dir / "basic_data.jsonl"
    with results_file.open("a", encoding="utf-8") as f:
        json.dump(results, f)
        f.write("\n")


def main(args: List[str]) -> None:
    run(parse_args(args))


if __name__ == "__main__":
    main(sys.argv[1:])
