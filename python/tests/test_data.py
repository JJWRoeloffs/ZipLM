from zip_lm.readdata import (
    BlimpPyItem,
    PyDataItems,
    get_blimp_data,
    get_sentence_corpus,
    get_data_items,
)


def test_get_blimpdata():
    data = get_blimp_data("blimp/data")
    assert all(isinstance(item, BlimpPyItem) for item in data)
    assert data[0].pair_id == 0


def test_blimpitem():
    item = BlimpPyItem(
        sentence_good="Some turtles alarm Kimberly",
        ll_sentence_good=0,
        sentence_bad="Some turtles come here Kimberley",
        ll_sentence_bad=0,
        field="syntax",
        linguistics_term="argument_structure",
        uid="transitive",
        pair_id=0,
    )
    assert item.sentence_good == "Some turtles alarm Kimberly"
    assert item.sentence_bad == "Some turtles come here Kimberley"
    assert item.field == "syntax"
    assert item.linguistics_term == "argument_structure"
    assert item.uid == "transitive"
    assert item.pair_id == 0

    item.sentence_good = "Blah"
    assert item.sentence_good == "Blah"


def test_get_sentence_corpus():
    corpus = get_sentence_corpus("data/train_10M")
    assert all(isinstance(sentence, str) for sentence in corpus.items())


def test_get_data_items():
    items = get_data_items("data/train_10M")
    assert all(isinstance(sentence, str) for sentence in items.bnc_spoken)
    assert all(isinstance(sentence, str) for sentence in items.childes)
    assert all(isinstance(sentence, str) for sentence in items.gutenberg)
    assert all(isinstance(sentence, str) for sentence in items.subtitiles)
    assert all(isinstance(sentence, str) for sentence in items.simple_wiki)
    assert all(isinstance(sentence, str) for sentence in items.switchboard)


def test_datitems():
    items = PyDataItems(
        ["This is a test sentence", "This is another test sentence"],
        ["This is a test sentence", "This is another test sentence"],
        ["This is a test sentence", "This is another test sentence"],
        ["This is a test sentence", "This is another test sentence"],
        ["This is a test sentence", "This is another test sentence"],
        ["This is a test sentence", "This is another test sentence"],
    )
    assert items.bnc_spoken[0] == "This is a test sentence"
