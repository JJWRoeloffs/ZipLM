from zip_lm.readdata import BlimpPyItem, get_blimp_data, get_sentence_corpus


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
