import sys
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

import nltk
from matplotlib import pyplot as plt

from zip_lm.readdata import get_data_items

from typing import List, Literal


@dataclass
class Args:
    data: Literal["10M", "100M"]
    cap: int
    imgpath: str


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
        "-f",
        "--file",
        help="The file to save the plot to",
        default="report/figures/zipf.png",
        nargs="?",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--cap",
        help="The top n words to get",
        type=int,
        default=1000,
        nargs="?",
        required=False,
    )
    arguments = parser.parse_args(args)
    return Args(arguments.data, arguments.cap, arguments.file)


def get_zipf(data: List[str], cap: int = 1000) -> List[float]:
    sents = [s for i in data for s in nltk.sent_tokenize(i)]
    # Yes, you have to pass .lower(): https://www.nltk.org/api/nltk.probability.FreqDist.html
    tokens = [t.lower() for s in sents for t in nltk.word_tokenize(s) if t.isalpha()]

    # Who needs NLTK.FreqDist when you have Counter in std? :P
    counts = Counter(tokens).most_common(cap)
    return [count / len(data) for _, count in counts]


def run(args: Args) -> None:
    data_dir = Path() / "data" / f"train_{args.data}"
    data = get_data_items(str(data_dir))

    xaxis = list(range(1, args.cap + 1))
    # The MandelBrot shift to more closly map natural language
    # (Mandelbrot, 1953, 1962)
    yaxis_expeced = [1 / (r + 2.7) for r in xaxis]

    plt.plot(xaxis, get_zipf(data.bnc_spoken, args.cap), label="bnc_spoken")
    plt.plot(xaxis, get_zipf(data.childes, args.cap), label="childes")
    plt.plot(xaxis, get_zipf(data.gutenberg, args.cap), label="gutenberg")
    plt.plot(xaxis, get_zipf(data.subtitiles, args.cap), label="subtitiles")
    plt.plot(xaxis, get_zipf(data.simple_wiki, args.cap), label="simple_wiki")
    plt.plot(xaxis, get_zipf(data.switchboard, args.cap), label="switchboard")
    plt.plot(xaxis, yaxis_expeced, label="pure zipf")

    plt.title(f"Frequency of the most common {args.cap} words of the dataset")
    plt.ylabel("Word Frequency")
    plt.xlabel("Word Rank")
    plt.legend()

    plt.yscale("log")
    plt.savefig(args.imgpath)


def main(args: List[str]) -> None:
    run(parse_args(args))


if __name__ == "__main__":
    main(sys.argv[1:])
