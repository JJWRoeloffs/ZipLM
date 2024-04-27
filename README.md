# Baby LM challange

This includes my not-really submission for [the BabyLM challange](https://babylm.github.io/), written for the ("extra challange") replacement exercice for the _vu Amstersam_ course Natural Language Processing.

The challange is to train a Language Model on an amount of data that is realistic for what a 13-year-old child would have seen in their life, where models must be able to assign a (pseudo) log-likelihood to a string of text and must be able to be fine-tuned to perform classification tasks.

### Running the project

The project is developed on python 3.11 and Rust 1.75, so, to run, install both on your system. After that, you should create a virtual envoirement, activate it, and install [maturin](https://pypi.org/project/maturin/) trough pip. After this, you can instll the project to your venv by running `maturin develop`
