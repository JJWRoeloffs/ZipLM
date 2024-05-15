use pyo3::prelude::*;

pub mod ngrams;
pub(crate) mod python;
pub mod readdata;
pub mod utils;

#[pymodule]
fn zip_lm(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    readdata::python::register_readdata_module(m)?;
    utils::python::register_utils_module(m)?;
    ngrams::python::register_ngrams_module(m)?;
    Ok(())
}
