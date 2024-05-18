pub mod python {
    #![allow(unused)]
    use crate::ngrams::python::PyNGramModel;
    use crate::readdata::blimp::python::BlimpPyItem;
    use crate::utils::LanguageModel;
    use crate::zipmodels::python::{PyBootstrapZipModel, PySoftmaxZipModel};
    use pyo3::prelude::*;
    use rayon::prelude::*;

    fn get_ll<T: LanguageModel>(model: &T, item: BlimpPyItem) -> BlimpPyItem {
        BlimpPyItem {
            ll_sentence_good: model.get_log_likelyhood_sentence(&item.sentence_good),
            ll_sentence_bad: model.get_log_likelyhood_sentence(&item.sentence_bad),
            ..item
        }
    }

    // You cannot expose generics to python, as Rust generics are compile-time.
    // I could try something with `dyn`, but I think just copy-paste is easier.
    #[pyfunction]
    fn get_ngrams_lls(model: &PyNGramModel, data: Vec<BlimpPyItem>) -> Vec<BlimpPyItem> {
        data.into_par_iter()
            .map(|item| get_ll(&model.inner, item))
            .collect()
    }
    #[pyfunction]
    fn get_bootstrap_lls(model: &PyBootstrapZipModel, data: Vec<BlimpPyItem>) -> Vec<BlimpPyItem> {
        data.into_par_iter()
            .map(|item| get_ll(&model.inner, item))
            .collect()
    }
    #[pyfunction]
    fn get_softmax_lls(model: &PySoftmaxZipModel, data: Vec<BlimpPyItem>) -> Vec<BlimpPyItem> {
        data.into_par_iter()
            .map(|item| get_ll(&model.inner, item))
            .collect()
    }

    pub(crate) fn register_eval_utils_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let eval_mod = PyModule::new_bound(parent.py(), "eval_utils")?;
        eval_mod.add_function(wrap_pyfunction!(get_ngrams_lls, &eval_mod)?)?;
        eval_mod.add_function(wrap_pyfunction!(get_bootstrap_lls, &eval_mod)?)?;
        eval_mod.add_function(wrap_pyfunction!(get_softmax_lls, &eval_mod)?)?;

        crate::python::add_submodule(parent, &eval_mod)?;
        Ok(())
    }
}
