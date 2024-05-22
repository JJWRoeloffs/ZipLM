use std::collections::HashMap;

use crate::utils::{Corpus, LanguageModel};
use std::f64::NEG_INFINITY;

// Pretty simple tokenizer for the N-grams. It's only a quick benchmark.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Token {
    Lemma(String),
    Unknown,
    Boundry,
}

impl Token {
    pub fn tokenize(sentence: String) -> Vec<Self> {
        sentence
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .flat_map(|c| c.to_lowercase())
            .collect::<String>()
            .split_whitespace()
            .map(|lemma| Token::Lemma(lemma.to_owned()))
            .collect()
    }

    pub fn tokenize_corpus(corpus: Corpus<Vec<String>>) -> Corpus<Vec<Vec<Self>>> {
        let mut items = corpus
            .items
            .into_iter()
            .map(Token::tokenize)
            .collect::<Vec<Vec<Self>>>();

        // Set any item with a count of less than 3 to Unknown.
        // Which is not as good as a valid vocabulary, but it might just work.
        // We need two more passes over the entire data for this, but that's why this is Rust.
        let mut counts: HashMap<Self, usize> = HashMap::new();
        for sentence in items.iter() {
            for token in sentence.iter() {
                // We cannot use .entry(token).or_insert(), as that requires to own the tokens.
                if let Some(count) = counts.get_mut(token) {
                    *count += 1
                } else {
                    counts.insert(token.clone(), 1);
                }
            }
        }
        for sentence in items.iter_mut() {
            for token in sentence.iter_mut() {
                let count = counts.get(&token).unwrap_or(&0);
                if *count < 3 {
                    *token = Token::Unknown;
                }
            }
        }

        Corpus { items }
    }

    pub fn pad_sentence(sentence: &mut Vec<Token>, count: usize) {
        sentence.splice(0..0, vec![Self::Boundry; count]);
    }
}

// Using indexes is not the most idiomatic rust,
// But it looks like the most effective solution.
#[derive(Debug, Default, PartialEq, Eq, Hash, Clone, Copy)]
struct Coord {
    sent_ind: usize,
    word_ind: usize,
}

impl Coord {
    fn new(sent_ind: usize, word_ind: usize) -> Self {
        Coord { sent_ind, word_ind }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct NGramModel {
    data: Vec<Vec<Token>>,
    lookup: HashMap<Token, Vec<Coord>>,
    n: usize,
    total_len: usize,
}

impl NGramModel {
    pub fn new(mut data: Vec<Vec<Token>>, n: usize) -> Self {
        let total_len = data.iter().map(|s| s.len()).sum();
        assert!(n >= 1, "Cannot make n-gram with n < 1");
        for sentence in data.iter_mut() {
            Token::pad_sentence(sentence, n - 1)
        }

        // Creating the lookup table last,
        // as, that way, the code doesn't fail when called on a boundry or unknown.
        let mut lookup: HashMap<Token, Vec<Coord>> = HashMap::new();
        for (sent_ind, &ref sentence) in data.iter().enumerate() {
            for (word_ind, &ref token) in sentence.iter().enumerate() {
                let coord = Coord::new(sent_ind, word_ind);
                // We cannot use .entry(token).or_insert(), as that requires to own the tokens.
                if let Some(coords) = lookup.get_mut(token) {
                    coords.push(coord);
                } else {
                    lookup.insert(token.clone(), vec![coord]);
                }
            }
        }

        Self {
            data,
            lookup,
            n,
            total_len,
        }
    }

    pub fn vocab_contains(&self, token: &Token) -> bool {
        self.lookup.contains_key(token)
    }

    pub fn sanitize(&self, sentence: Vec<Token>) -> Vec<Token> {
        sentence
            .into_iter()
            .map(|token| match self.vocab_contains(&token) {
                true => token,
                false => Token::Unknown,
            })
            .collect()
    }

    fn get_count(&self, ngram: &[Token]) -> usize {
        assert!(ngram.len() >= self.n - 1, "Cannot count non n-gram");
        assert!(
            ngram.iter().all(|token| self.vocab_contains(token)),
            "Cannot get the count of unknown tokens. Replace those with Token::Unknown"
        );
        let target = ngram.last().unwrap();
        if target == &Token::Boundry {
            return self.data.len();
        }
        let mut count: usize = 0;
        for case in self.lookup.get(&target).unwrap() {
            let start_index = case.word_ind - (ngram.len() - 1);
            if self.data[case.sent_ind][start_index..].starts_with(ngram) {
                count += 1;
            }
        }
        count
    }

    fn get_ll_many(&self, sentence: &Vec<Token>) -> f64 {
        // This uses the common approximation that
        // P(W1, W2, ..., Wn) ~ C(W1, W2, ..., Wn) / C(W2, ..., Wn)
        // Technically, this calculation doesn't allow the model to predict the end token
        // (Meaning suddenly stopped sentences aren't penalized,)
        // But chaning that would be too much effort for now.
        let mut padded_sentence = sentence.clone();
        Token::pad_sentence(&mut padded_sentence, self.n);
        let mut res: f64 = 0.0;
        for i in 0..sentence.len() {
            let top = self.get_count(&padded_sentence[i..i + self.n]) as f64;
            let bot = self.get_count(&padded_sentence[i..i + self.n - 1]) as f64;
            res += (top / bot).log10()
        }
        res
    }

    fn get_ll_one(&self, sentence: &Vec<Token>) -> f64 {
        let mut res: f64 = 0.0;
        for token in sentence {
            if let Some(count) = self.lookup.get(&token) {
                res += (count.len() as f64 / self.total_len as f64).log10();
            } else {
                return NEG_INFINITY;
            }
        }
        res
    }

    pub fn get_log_likelyhood(&self, sentence: &Vec<Token>) -> f64 {
        match self.n {
            1 => self.get_ll_one(sentence),
            _ => self.get_ll_many(sentence),
        }
    }
}

impl LanguageModel for NGramModel {
    fn get_log_likelyhood_sentence(&self, sentence: &str) -> f64 {
        let sentence = self.sanitize(Token::tokenize(sentence.to_owned()));
        self.get_log_likelyhood(&sentence)
    }
}

pub mod python {
    #![allow(unused)]
    use super::*;
    use crate::utils::python::*;
    use pyo3::{exceptions::PyValueError, prelude::*};

    #[pyclass]
    #[derive(Debug, PartialEq, Eq, Clone)]
    pub enum PyTokenKind {
        Lemma = 0,
        Boundry = 1,
        Unknown = 2,
    }
    #[pymethods]
    impl PyTokenKind {
        #[new]
        fn new(from: usize) -> PyResult<Self> {
            match from {
                0 => Ok(Self::Lemma),
                1 => Ok(Self::Boundry),
                2 => Ok(Self::Unknown),
                _ => Err(PyValueError::new_err("{from} is not a PyTokenKind")),
            }
        }
    }
    #[pyfunction]
    pub fn tokenkind_from_str(string: String) -> PyResult<PyTokenKind> {
        match string.as_str() {
            "lemma" => Ok(PyTokenKind::Lemma),
            "boundry" => Ok(PyTokenKind::Boundry),
            "unknown" => Ok(PyTokenKind::Unknown),
            _ => Err(PyValueError::new_err("{string} is not a Token Kind")),
        }
    }

    #[pyclass(get_all)]
    #[derive(Debug, PartialEq, Eq, Clone)]
    pub struct PyToken {
        pub kind: PyTokenKind,
        pub data: String,
    }
    #[pymethods]
    impl PyToken {
        #[new]
        fn new(kind: PyTokenKind, data: String) -> PyResult<Self> {
            match (&kind, data.as_str()) {
                (PyTokenKind::Boundry, "") => Ok(Self { kind, data }),
                (PyTokenKind::Unknown, "") => Ok(Self { kind, data }),
                (PyTokenKind::Lemma, string) if string.chars().all(|c| c.is_alphanumeric()) => {
                    Ok(Self { kind, data })
                }
                (_, _) => Err(PyValueError::new_err("{data} is not valid token data")),
            }
        }
    }
    impl PyToken {
        fn new_rs(kind: PyTokenKind, data: String) -> Self {
            Self { kind, data }
        }
        #[inline(always)]
        fn from_token(token: Token) -> Self {
            match token {
                Token::Lemma(data) => Self::new_rs(PyTokenKind::Lemma, data),
                Token::Unknown => Self::new_rs(PyTokenKind::Unknown, "".to_owned()),
                Token::Boundry => Self::new_rs(PyTokenKind::Boundry, "".to_owned()),
            }
        }
        #[inline(always)]
        fn to_token(self) -> Token {
            match self.kind {
                PyTokenKind::Lemma => Token::Lemma(self.data),
                PyTokenKind::Boundry => Token::Boundry,
                PyTokenKind::Unknown => Token::Unknown,
            }
        }
    }

    create_pycorpus!(PyCorpusTokens, Vec<Vec<PyToken>>);

    #[pyfunction]
    pub fn tokenize_pycorpus(sentences: &PyCorpusSentences) -> PyCorpusTokens {
        let tokens = Token::tokenize_corpus(sentences.inner.clone());
        let pytokens = tokens
            .items
            .into_iter()
            .map(|inner| inner.into_iter().map(PyToken::from_token).collect())
            .collect::<Vec<Vec<PyToken>>>();

        PyCorpusTokens::new(pytokens)
    }

    #[pyfunction]
    pub fn pytokenize(sentence: String) -> Vec<PyToken> {
        Token::tokenize(sentence)
            .into_iter()
            .map(PyToken::from_token)
            .collect()
    }

    #[pyclass]
    #[derive(Debug, PartialEq, Eq, Clone)]
    pub struct PyNGramModel {
        pub inner: NGramModel,
    }

    #[pymethods]
    impl PyNGramModel {
        #[new]
        pub fn new(data: &PyCorpusTokens, n: usize) -> Self {
            let tokens = data
                .inner
                .clone()
                .items
                .into_iter()
                .map(|inner| inner.into_iter().map(PyToken::to_token).collect())
                .collect::<Vec<Vec<Token>>>();

            Self {
                inner: NGramModel::new(tokens, n),
            }
        }

        pub fn sanitize(&self, data: Vec<PyToken>) -> Vec<PyToken> {
            data.into_iter()
                .map(
                    |token| match self.inner.vocab_contains(&token.clone().to_token()) {
                        true => token,
                        false => PyToken::new_rs(PyTokenKind::Unknown, "".into()),
                    },
                )
                .collect()
        }

        pub fn get_log_likelyhood(&self, data: Vec<PyToken>) -> f64 {
            let tokens = data.into_iter().map(PyToken::to_token).collect();
            self.inner.get_log_likelyhood(&tokens)
        }
    }

    pub(crate) fn register_ngrams_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let ngrams_mod = PyModule::new_bound(parent.py(), "ngrams")?;
        ngrams_mod.add_class::<PyToken>()?;
        ngrams_mod.add_class::<PyTokenKind>()?;
        ngrams_mod.add_class::<PyCorpusTokens>()?;
        ngrams_mod.add_function(wrap_pyfunction!(tokenkind_from_str, &ngrams_mod)?)?;
        ngrams_mod.add_function(wrap_pyfunction!(tokenize_pycorpus, &ngrams_mod)?)?;
        ngrams_mod.add_function(wrap_pyfunction!(pytokenize, &ngrams_mod)?)?;
        ngrams_mod.add_class::<PyNGramModel>()?;

        crate::python::add_submodule(parent, &ngrams_mod)?;
        Ok(())
    }
}
