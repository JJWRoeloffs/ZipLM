import baby_lm


def test_get_blimpdata():
    data = baby_lm.readdata.get_blimp_data("blimp/data")
    assert all(isinstance(item, baby_lm.readdata.BlimpPyItem) for item in data)
    assert data[0].pair_id == 0


def test_blimpitem():
    item = baby_lm.readdata.BlimpPyItem(
        sentence_good="Some turtles alarm Kimberly",
        sentence_bad="Some turtles come here Kimberley",
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
    corpus = baby_lm.readdata.get_sentence_corpus("data/train_10M")
    assert all(isinstance(sentence, str) for sentence in corpus.items())
