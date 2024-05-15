use pyo3::prelude::*;

pub mod ngrams;
pub mod readdata;
pub mod utils;

#[pymodule]
fn baby_lm(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    readdata::python::register_readdata_module(m)?;
    utils::python::register_utils_module(m)?;
    ngrams::python::register_ngrams_module(m)?;
    Ok(())
}

// See the following issues as to why this is needed:
// * https://github.com/PyO3/pyo3/issues/759
// * https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
// This'll hopefully be fixed in 0.22: https://github.com/PyO3/pyo3/issues/3900
pub(crate) fn add_submodule(
    parent: &Bound<'_, PyModule>,
    child: &Bound<'_, PyModule>,
) -> PyResult<()> {
    parent.add_submodule(child)?;
    parent
        .py()
        .import_bound("sys")?
        .getattr("modules")?
        // parent.name()? doesn't work, as that would be `baby_lm.baby_lm`
        .set_item(format!("baby_lm.{}", child.name()?), child)?;
    Ok(())
}
