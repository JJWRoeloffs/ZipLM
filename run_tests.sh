#!/usr/bin/env bash

python -m zip_lm.scripts.evaluate_random 42 100

python -m zip_lm.scripts.evaluate_ngrams 42 100 1 -d 10M
python -m zip_lm.scripts.evaluate_ngrams 42 100 2 -d 10M

python -m zip_lm.scripts.evaluate_bootstrapzipmodel 42 100 1000 1000 -t threequarterq -d 10M
python -m zip_lm.scripts.evaluate_bootstrapzipmodel 42 100 1000 1000 -t halfq -d 10M
python -m zip_lm.scripts.evaluate_bootstrapzipmodel 42 100 1000 1000 -t quarterq -d 10M

python -m zip_lm.scripts.evaluate_softmaxzipmodel 42 100 10 -t threequarterq -d 10M
python -m zip_lm.scripts.evaluate_softmaxzipmodel 42 100 10 -t halfq -d 10M
python -m zip_lm.scripts.evaluate_softmaxzipmodel 42 100 10 -t quarterq -d 10M

python -m zip_lm.scripts.evaluate_t5 42 100 --model_name t5-3b
