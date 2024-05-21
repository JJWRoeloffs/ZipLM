from datasets import DatasetDict
from transformers import T5TokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from typing import Iterator, List, Tuple


def batch_iter(*items: List[str], batch_size: int = 1000) -> Iterator[List[str]]:
    for item in items:
        for i in range(0, len(item), batch_size):
            yield item[i : i + batch_size]


def make_tokenizer(training_data: Iterator[List[str]]) -> T5TokenizerFast:
    # This should be the T5 setup. I thought I could just download the config somehow,
    # But did not find that.
    tokenizer = Tokenizer(models.Unigram())  # type: ignore
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Lowercase(),
        ]
    )  # type: ignore
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()  # type: ignore
    trainer = trainers.UnigramTrainer(
        vocab_size=25000,
        special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"],
        unk_token="<unk>",
    )
    tokenizer.train_from_iterator(training_data, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )  # type: ignore

    tokenizer.decoder = decoders.Metaspace()  # type: ignore
    return T5TokenizerFast(tokenizer_object=tokenizer)


def get_tokenizer(
    data: DatasetDict, batch_size: int = 1000
) -> Tuple[T5TokenizerFast, int]:
    items = data["train"]["text"]
    tokenizer = make_tokenizer(batch_iter(items, batch_size=batch_size))
    # Yes, we tokenize twice. No, I don't care.
    tokenized_texts = [tokenizer.encode(s, add_special_tokens=True) for s in items]
    max_seq_length = max(len(s) for s in tokenized_texts)
    return tokenizer, min(max_seq_length, tokenizer.model_max_length)
