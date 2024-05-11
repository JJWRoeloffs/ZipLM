// When writing application code, you rarely care what the error is, just that one happened.
// I could also just return Option everywhere, but I like this better.
pub type Res<T> = Result<T, ()>;

// The most simple type of corpus there could be, just a list of sentences.
#[derive(Debug, Default, Clone)]
pub struct Corpus<T: Default + Clone> {
    pub items: T,
}

pub mod python {
    #![allow(unused)]
    use super::*;
    use pyo3::prelude::*;

    macro_rules! create_pycorpus {
        ($name: ident, $type: ty) => {
            #[pyclass]
            pub struct $name {
                pub inner: Corpus<$type>,
            }
            #[pymethods]
            impl $name {
                #[new]
                pub fn new(items: $type) -> Self {
                    Self {
                        inner: Corpus { items },
                    }
                }
                pub fn items(&self) -> $type {
                    self.inner.items.clone()
                }
            }
        };
    }
    pub(crate) use create_pycorpus;

    create_pycorpus!(PyCorpusSentences, Vec<String>);

    pub fn register_utils_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let utils_mod = PyModule::new_bound(parent.py(), "utils")?;
        utils_mod.add_class::<PyCorpusSentences>()?;

        parent.add_submodule(&utils_mod)?;
        Ok(())
    }
}
