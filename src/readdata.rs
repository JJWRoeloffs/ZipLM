use crate::utils::Res;

use std::fs;
use std::path::Path;

pub mod blimp;
pub mod testdata;

#[inline(always)]
pub fn get_file_contents(path: &Path) -> Res<String> {
    fs::read_to_string(path)
        .map_err(|err| eprintln!("{err} | Could not read from file {}", path.display()))
}
