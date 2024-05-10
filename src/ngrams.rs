use std::collections::HashMap;

use crate::utils::Corpus;

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
            .collect::<String>()
            .split_whitespace()
            .map(str::to_owned)
            .map(|lemma| Token::Lemma(lemma))
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

#[derive(Debug)]
pub struct NGramModel {
    data: Vec<Vec<Token>>,
    lookup: HashMap<Token, Vec<Coord>>,
    n: usize,
}

impl NGramModel {
    pub fn new(mut data: Vec<Vec<Token>>, n: usize) -> Self {
        assert!(n >= 2, "Cannot make n-gram with n < 2");
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

        Self { data, lookup, n }
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

    pub fn get_log_likelyhood(&self, sentence: &Vec<Token>) -> f64 {
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
}
