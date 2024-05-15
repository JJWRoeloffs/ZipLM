use pyo3::prelude::*;

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
        // parent.name()? doesn't work, as that would be `zip_lm.zip_lm`
        .set_item(format!("zip_lm.{}", child.name()?), child)?;
    Ok(())
}
