use serde::Deserialize;
use serde_aux::prelude::*;
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
    #[serde(
        rename(deserialize = "pairID"),
        deserialize_with = "deserialize_number_from_string"
    )]
    pub pair_id: usize,
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

pub mod python {
    #![allow(unused)]
    use super::*;
    use pyo3::{exceptions::PyIOError, prelude::*};

    impl BlimpItem {
        fn to_python(self) -> BlimpPyItem {
            BlimpPyItem {
                sentence_good: self.sentence_good,
                ll_sentence_good: 0.0,
                sentence_bad: self.sentence_bad,
                ll_sentence_bad: 0.0,
                field: self.field,
                linguistics_term: self.linguistics_term,
                uid: self.uid,
                pair_id: self.pair_id,
            }
        }
    }

    // used for both reading data, and evaluation (with the log likelyhood values.)
    #[derive(Debug, Clone)]
    #[pyclass(get_all, set_all)]
    pub struct BlimpPyItem {
        pub sentence_good: String,
        pub ll_sentence_good: f64,
        pub sentence_bad: String,
        pub ll_sentence_bad: f64,
        pub field: String,
        pub linguistics_term: String,
        pub uid: String,
        pub pair_id: usize,
    }

    #[pymethods]
    impl BlimpPyItem {
        #[new]
        fn new(
            sentence_good: String,
            ll_sentence_good: f64,
            sentence_bad: String,
            ll_sentence_bad: f64,
            field: String,
            linguistics_term: String,
            uid: String,
            pair_id: usize,
        ) -> Self {
            Self {
                sentence_good,
                ll_sentence_good,
                sentence_bad,
                ll_sentence_bad,
                field,
                linguistics_term,
                uid,
                pair_id,
            }
        }
    }

    #[pyfunction]
    pub fn get_blimp_data(path: String) -> PyResult<Vec<BlimpPyItem>> {
        Ok(read_blimpitems_from_dir(Path::new(&path))
            .map_err(|_| PyIOError::new_err(format!("Could not read from {path}")))?
            .into_iter()
            .map(BlimpItem::to_python)
            .collect())
    }
}
