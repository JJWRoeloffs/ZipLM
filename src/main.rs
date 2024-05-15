use std::path::Path;

mod readdata;
use readdata::blimp;
use readdata::testdata::DataItems;

mod utils;
use utils::Res;

mod ngrams;
use ngrams::{NGramModel, Token};

mod python;

fn main() -> Res<()> {
    // let mut _train100 = DataItems::from_dir(Path::new("data/train_100M"))?;
    let train10 = DataItems::from_dir(Path::new("data/train_100M"))?.to_corpus();
    println!("test0");
    let tokens = Token::tokenize_corpus(train10);
    println!("test1");
    let ngram_model = NGramModel::new(tokens.items, 2);
    println!("test2");

    let test_sentence1 =
        ngram_model.sanitize(Token::tokenize("This is a test sentence".to_owned()));
    let test_sentence2 =
        ngram_model.sanitize(Token::tokenize("This is another test sentence".to_owned()));
    let test_sentence3 = ngram_model.sanitize(Token::tokenize(
        "This is another, better test sentence".to_owned(),
    ));

    let base: f64 = 10.0;

    dbg!(base.powf(ngram_model.get_log_likelyhood(&test_sentence1)));
    dbg!(base.powf(ngram_model.get_log_likelyhood(&test_sentence2)));
    dbg!(base.powf(ngram_model.get_log_likelyhood(&test_sentence3)));

    // let mut _test = DataItems::from_dir(Path::new("data/test"))?;

    // let mut _blimps = blimp::read_blimpitems_from_dir(Path::new("blimp/data"))?;
    Ok(())
}
