use std::cmp::Ordering;

// When writing application code, you rarely care what the error is, just that one happened.
// I could also just return Option everywhere, but I like this better.
pub type Res<T> = Result<T, ()>;

// The most simple type of corpus there could be, just a list of sentences.
#[derive(Debug, Default, Clone)]
pub struct Corpus<T: Default + Clone> {
    pub items: T,
}

#[derive(PartialEq, PartialOrd, Debug, Default, Clone, Copy)]
pub struct NonNanF64(f64);
impl NonNanF64 {
    pub fn new(val: f64) -> Option<NonNanF64> {
        match val.is_nan() {
            true => None,
            false => Some(Self(val)),
        }
    }
    pub fn value(&self) -> f64 {
        self.0
    }
    pub fn mean(vals: Vec<Self>) -> Self {
        Self::new(vals.iter().map(|x| x.0).sum::<f64>() / vals.len() as f64)
            .expect("Mean does not create nan")
    }
    pub fn quantile(vals: Vec<Self>, quant: f64) -> Self {
        let mut tmp = vals.clone();
        tmp.sort_unstable();
        tmp[(vals.len() as f64 * quant) as usize]
    }
}
impl Eq for NonNanF64 {}
impl Ord for NonNanF64 {
    fn cmp(&self, other: &NonNanF64) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub mod python {
    #![allow(unused)]
    use super::*;
    use pyo3::prelude::*;

    macro_rules! create_pycorpus {
        ($name: ident, $type: ty) => {
            #[pyclass]
            #[derive(Debug, Clone)]
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

    pub(crate) fn register_utils_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let utils_mod = PyModule::new_bound(parent.py(), "utils")?;
        utils_mod.add_class::<PyCorpusSentences>()?;

        crate::python::add_submodule(parent, &utils_mod)?;
        Ok(())
    }
}
