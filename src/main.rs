#![allow(unused)]

use std::path::Path;

mod readdata;
use readdata::blimp;
use readdata::testdata::DataItems;

mod utils;
use utils::{NonNanF64, Res};

mod ngrams;
use ngrams::{NGramModel, Token};
use zipmodels::BootstrapZipModel;

use crate::zipmodels::SoftmaxZipModel;

mod python;

mod zipmodels;

fn run_ngrams_model() -> Res<()> {
    let train10 = DataItems::from_dir(Path::new("data/train_100M"))?.to_corpus();
    let tokens = Token::tokenize_corpus(train10);
    let ngram_model = NGramModel::new(tokens.items, 2);

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
    Ok(())
}

fn run_bootstrapzip_model() -> Res<()> {
    let train10 = DataItems::from_dir(Path::new("data/train_10M"))?.to_corpus();
    let data = train10
        .items
        .into_iter()
        .map(|sent| sent.into_bytes())
        .collect::<Vec<Vec<u8>>>();

    let mut rng = rand::thread_rng();

    let t = |xs| NonNanF64::quantile(xs, 0.75);
    let bootstrap_zip_model = BootstrapZipModel::new(data, 1000, 1000, t, &mut rng);

    let test_sentence1 = "This is a test sentence".as_bytes();

    let base: f64 = 10.0;
    dbg!(base.powf(bootstrap_zip_model.get_log_likelyhood(&test_sentence1)));

    Ok(())
}

fn run_softmaxzip_model() -> Res<()> {
    let train10 = DataItems::from_dir(Path::new("data/train_10M"))?.to_corpus();
    let data = train10
        .items
        .into_iter()
        .map(|sent| sent.into_bytes())
        .collect::<Vec<Vec<u8>>>();

    let mut rng = rand::thread_rng();

    let t = |xs| NonNanF64::quantile(xs, 0.75);
    let bootstrap_zip_model = SoftmaxZipModel::new(data, 10, t, &mut rng);

    let test_sentence1 = "This is a test sentence".as_bytes();

    let base: f64 = 10.0;
    dbg!(base.powf(bootstrap_zip_model.get_log_likelyhood(&test_sentence1)));

    Ok(())
}

fn main() -> Res<()> {
    run_ngrams_model()?;
    run_softmaxzip_model()?;
    run_bootstrapzip_model()?;

    Ok(())
}
