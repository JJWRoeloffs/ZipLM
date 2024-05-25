# A gzip-based Language Model

This repository contains my not-really submission for [the BabyLM challenge](https://babylm.github.io/), written for the ("extra challenge") replacement exercise for the _vu Amstersam_ course Natural Language Processing.

The challenge is to train a Language Model on an amount of data that is realistic for what a 13-year-old child would have seen in their life, where models must be able to assign a (pseudo) log-likelihood to a string of text and must be able to be fine-tuned to perform classification tasks.

For this project, I decided to try to replicate the work of [Jiang et al.](https://doi.org/10.48550/arXiv.2212.09410) and see if I could make a Language model that spits out log-likelihoods based entirely on gzip.

I came up with techniques for this, one making use of the Softmax function, and another making use of so-called bootstrap tests, but neither managed to perform better than an n-gram baseline.

## Running the project

The project is developed on Python 3.11 and Rust 1.75, so, to run, install both on your system. After that, you should create a virtual environment, activate it, and install [maturin](https://pypi.org/project/maturin/) through pip. After this, you can install the project to your venv by running `maturin develop` (Or, `maturin develop --release` if you want the models to take less than a lifetime to run.)

The Python dependencies (found in `pyproject.toml`) include `transformers`, `torch`, and `jax`, so installing those first in the way recommended for your OS and available CUDA compute is advisable if you're planning to run the T5 parts of the project.

The project is developed and tested on Linux. It working on any other system is accidental.

### Extracting the data

The data distributed for the BabyLM challenge can be found [here](https://osf.io/ad7qg/), and should all be extracted to `data/`, which is to say the code expects files like `data/train_10M/childes.train` to exist.

### Running the scripts

The project's interface is a series of Python scripts that can be called to retrieve certain test information. The most important tests are all combined in `run_tests.sh`, which appends to the results found under `results/`. However, there are a few more scripts available to run, for example, `get_data_metrics` to get basic metrics about the training set, `plot_word_usage`, to plot the word frequencies against the expected zipf curve, and `build_t5model` to train a t5 model from the training set.

## Building the report

The report is written in basic LaTeX with very few extensions, so on most systems with `pdflatex` installed, you should be able to run `make` inside the `report/` directory to compile a pdf.
