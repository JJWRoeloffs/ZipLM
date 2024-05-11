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

pub mod python {
    #![allow(unused)]
    use super::*;
    use pyo3::prelude::*;

    pub fn register_readdata_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        use blimp::python::*;
        use testdata::python::*;
        let data_mod = PyModule::new_bound(parent.py(), "readdata")?;
        data_mod.add_function(wrap_pyfunction!(get_blimp_data, &data_mod)?)?;
        data_mod.add_class::<BlimpPyItem>()?;
        data_mod.add_function(wrap_pyfunction!(get_sentence_corpus, &data_mod)?)?;

        parent.add_submodule(&data_mod)?;
        Ok(())
    }
}
