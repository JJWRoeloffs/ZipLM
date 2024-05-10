use serde::Deserialize;
use std::fs::{self, DirEntry};
use std::path::Path;

use super::get_file_contents;
use crate::utils::Res;

// There are sometimes more fields in the json, but that is allowed
// https://github.com/serde-rs/serde/pull/201
#[derive(Deserialize)]
pub struct BlimpItem {
    pub sentence_good: String,
    pub sentence_bad: String,
    pub field: String,
    pub linguistics_term: String,
    #[serde(rename(deserialize = "UID"))]
    pub uid: String,
    #[serde(rename(deserialize = "pairID"))]
    pub pair_id: String,
}

fn read_blimpitems_from_file(path: &Path) -> Res<Vec<BlimpItem>> {
    println!("Reading from: {}", path.display());
    get_file_contents(path)
        .map_err(|_| eprintln!("Could not read file: {}", path.display()))
        .iter()
        .flat_map(|items| {
            items.lines().map(|line| {
                serde_json::from_str::<BlimpItem>(line).map_err(|err| {
                    eprintln!("{err} | Could not read a line in file {}", path.display())
                })
            })
        })
        .collect()
}

pub fn read_blimpitems_from_dir(path: &Path) -> Res<Vec<BlimpItem>> {
    Ok(fs::read_dir(path)
        .map_err(|err| eprintln!("{err} | Could not read directory: {}", path.display()))?
        .into_iter()
        .collect::<Result<Vec<DirEntry>, std::io::Error>>() // Collecting to transpose
        .map_err(|err| eprintln!("{err} | While opening file in directory {}", path.display()))?
        .iter()
        .flat_map(|dir_entry| read_blimpitems_from_file(&dir_entry.path()))
        .flatten()
        .collect::<Vec<BlimpItem>>())
}
