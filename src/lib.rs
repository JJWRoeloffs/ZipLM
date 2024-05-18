use pyo3::prelude::*;

pub mod eval_utils;
pub mod ngrams;
pub(crate) mod python;
pub mod readdata;
pub mod utils;
pub mod zipmodels;

#[pymodule]
fn zip_lm(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    readdata::python::register_readdata_module(m)?;
    utils::python::register_utils_module(m)?;
    ngrams::python::register_ngrams_module(m)?;
    zipmodels::python::register_zipmodels_module(m)?;
    eval_utils::python::register_eval_utils_module(m)?;
    Ok(())
}
