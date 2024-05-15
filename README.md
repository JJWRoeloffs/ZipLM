# A gzip-based Language Model

This includes my not-really submission for [the BabyLM challange](https://babylm.github.io/), written for the ("extra challenge") replacement exercise for the _vu Amstersam_ course Natural Language Processing.

The challenge is to train a Language Model on an amount of data that is realistic for what a 13-year-old child would have seen in their life, where models must be able to assign a (pseudo) log-likelihood to a string of text and must be able to be fine-tuned to perform classification tasks.

For this project, I decided to try to replicate the work of [Jiang et al.](https://doi.org/10.48550/arXiv.2212.09410), and see if I could make a Language model that spits out log likelyhoods based entirely on gzip.

## Running the project

The project is developed on python 3.11 and Rust 1.75, so, to run, install both on your system. After that, you should create a virtual environment, activate it, and install [maturin](https://pypi.org/project/maturin/) trough pip. After this, you can install the project to your venv by running `maturin develop`

The project is developed and tested on Linux. It working on any other system is accidental.

### Extracting the data

The data distributed for the BabyLM challenge can be found [here](https://osf.io/ad7qg/), and should all be extracted to `data/`, which is to say the code expects files like `data/train_10M/childes.train` to exist.

## Building the report

The report is written in basic LaTeX with very few extensions, so on most systems with `pdflatex` installed, you should be able to run `pdflatex ./report/main.tex`
