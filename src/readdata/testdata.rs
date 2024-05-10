use crate::utils::{Corpus, Res};
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub struct DataItems<T> {
    pub bnc_spoken: T,
    pub childes: T,
    pub gutenberg: T,
    pub subtitiles: T,
    pub simple_wiki: T,
    pub switchboard: T,
}

impl DataItems<Vec<String>> {
    pub fn from_dir(path: &Path) -> Res<DataItems<Vec<String>>> {
        let items = read_dataitems_from_dir(path)?;
        Ok(DataItems {
            bnc_spoken: parse_corpus(&items.bnc_spoken, parse_bnc_spoken),
            childes: parse_corpus(&items.childes, parse_childes),
            gutenberg: parse_corpus(&items.gutenberg, parse_gutenberg),
            subtitiles: parse_corpus(&items.subtitiles, parse_open_subtitles),
            simple_wiki: parse_corpus(&items.simple_wiki, parse_simple_wiki),
            switchboard: parse_corpus(&items.switchboard, parse_switchboard),
        })
    }

    pub fn to_corpus(mut self) -> Corpus<Vec<String>> {
        let mut items = Vec::with_capacity(
            self.bnc_spoken.len()
                + self.childes.len()
                + self.gutenberg.len()
                + self.subtitiles.len()
                + self.simple_wiki.len()
                + self.switchboard.len(),
        );
        items.append(&mut self.bnc_spoken);
        items.append(&mut self.gutenberg);
        items.append(&mut self.subtitiles);
        items.append(&mut self.simple_wiki);
        items.append(&mut self.switchboard);

        Corpus { items }
    }
}

#[inline(always)]
fn get_file_contents(path: &Path) -> Res<String> {
    fs::read_to_string(path)
        .map_err(|err| eprintln!("{err} | Could not read from file {}", path.display()))
}

pub fn read_dataitems_from_dir(path: &Path) -> Res<DataItems<String>> {
    let mut data_items = DataItems {
        bnc_spoken: "".to_owned(),
        childes: "".to_owned(),
        gutenberg: "".to_owned(),
        subtitiles: "".to_owned(),
        simple_wiki: "".to_owned(),
        switchboard: "".to_owned(),
    };

    for entry in fs::read_dir(path)
        .map_err(|err| eprintln!("{err} | Could not read directory {}", path.display()))?
    {
        let entrypath = entry
            .map_err(|err| eprintln!("{err} | Could not read entry in {}", path.display()))?
            .path();
        match entrypath
            .file_stem()
            .ok_or_else(|| eprintln!("Directory entry is not a path"))?
            .to_str()
            .ok_or_else(|| eprintln!("Directory entry is not a valid string"))?
        {
            "bnc_spoken" => data_items.bnc_spoken = get_file_contents(&entrypath)?,
            "childes" => data_items.childes = get_file_contents(&entrypath)?,
            "gutenberg" => data_items.gutenberg = get_file_contents(&entrypath)?,
            "open_subtitles" => data_items.subtitiles = get_file_contents(&entrypath)?,
            "simple_wiki" => data_items.simple_wiki = get_file_contents(&entrypath)?,
            "switchboard" => data_items.switchboard = get_file_contents(&entrypath)?,
            _ => continue,
        }
        println!("Read: {}", entrypath.display());
    }
    Ok(data_items)
}

#[inline(always)]
fn parse_switchboard(s: &str) -> Option<&str> {
    // Removes "A:\t"
    s.get(3..)
}

#[inline(always)]
fn parse_simple_wiki(s: &str) -> Option<&str> {
    // Ignore empty strings, and strings starting with =
    match s.chars().next() {
        Some('=') => None,
        Some(_) => Some(s),
        None => None,
    }
}

#[inline(always)]
fn parse_open_subtitles(s: &str) -> Option<&str> {
    Some(s)
}

#[inline(always)]
fn parse_gutenberg(s: &str) -> Option<&str> {
    // Ignore all strings with not alphabetic chars (that includes empties),
    // and all strings of which all alphabetic chars are caps (e.g. "= = = PG45447 = = =")
    match s
        .chars()
        .filter(|c| c.is_alphabetic())
        .all(char::is_uppercase)
    {
        true => None,
        false => Some(s),
    }
}

#[inline(always)]
fn parse_childes(s: &str) -> Option<&str> {
    // Only get actual spoken data, not tokens surrounding
    match s.chars().next() {
        Some('*') => s.get(6..),
        Some(_) => None,
        None => None,
    }
}

#[inline(always)]
fn parse_bnc_spoken(s: &str) -> Option<&str> {
    Some(s)
}

#[inline(always)]
fn parse_corpus<F>(s: &str, fun: F) -> Vec<String>
where
    F: Fn(&str) -> Option<&str>,
{
    s.lines().filter_map(fun).map(String::from).collect()
}
