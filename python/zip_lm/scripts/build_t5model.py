import sys
import argparse

from zip_lm.t5 import build_model

from typing import List


def main(args: List[str]):
    parser = argparse.ArgumentParser(
        prog="build_t5model",
        description="The script to use to train a custom T5 Model",
    )
    parser.add_argument("--seed", type=int, help="The seed to use for the rng")
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        nargs="?",
        help="Ratio of tokens to mask for span masked language modeling loss",
    )
    parser.add_argument(
        "--mean_noise_span_length",
        type=float,
        default=3.0,
        nargs="?",
        help="Mean span length of masked tokens",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        nargs="?",
        help="The directory to save the model to",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        nargs="?",
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        nargs="?",
        help="Batch size per GPU/TPU core/CPU for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        nargs="?",
        help="Batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        nargs="?",
        help="The initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        nargs="?",
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        nargs="?",
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        nargs="?",
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        nargs="?",
        help="Weight decay for AdamW if we apply some.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        nargs="?",
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1,
        nargs="?",
        help="Run an evaluation every X steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1,
        nargs="?",
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--do_eval",
        type=int,
        default=True,
        nargs="?",
        help="Whether to run eval on the dev set.",
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

    build_model(
        mlm_probability=arguments.mlm_probability,
        mean_noise_span_length=arguments.mean_noise_span_length,
        seed=arguments.seed,
        output_dir=arguments.output_dir,
        num_train_epochs=arguments.num_train_epochs,
        per_device_train_batch_size=arguments.per_device_train_batch_size,
        per_device_eval_batch_size=arguments.per_device_eval_batch_size,
        learning_rate=arguments.learning_rate,
        warmup_steps=arguments.warmup_steps,
        adam_beta1=arguments.adam_beta1,
        adam_beta2=arguments.adam_beta2,
        weight_decay=arguments.weight_decay,
        logging_steps=arguments.logging_steps,
        eval_steps=arguments.eval_steps,
        save_steps=arguments.save_steps,
        do_eval=arguments.do_eval,
        data=arguments.data,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
