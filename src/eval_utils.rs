pub mod python {
    #![allow(unused)]
    use crate::ngrams::python::PyNGramModel;
    use crate::readdata::blimp::python::BlimpPyItem;
    use crate::utils::LanguageModel;
    use crate::zipmodels::python::{PyBootstrapZipModel, PySoftmaxZipModel};
    use pyo3::prelude::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_pcg::Pcg32;
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
    pub fn get_ngrams_lls(model: &PyNGramModel, data: Vec<BlimpPyItem>) -> Vec<BlimpPyItem> {
        data.into_par_iter()
            .map(|item| get_ll(&model.inner, item))
            .collect()
    }
    #[pyfunction]
    pub fn get_bootstrap_lls(
        model: &PyBootstrapZipModel,
        data: Vec<BlimpPyItem>,
    ) -> Vec<BlimpPyItem> {
        data.into_par_iter()
            .map(|item| get_ll(&model.inner, item))
            .collect()
    }
    #[pyfunction]
    pub fn get_softmax_lls(model: &PySoftmaxZipModel, data: Vec<BlimpPyItem>) -> Vec<BlimpPyItem> {
        data.into_par_iter()
            .map(|item| get_ll(&model.inner, item))
            .collect()
    }

    #[pyfunction]
    pub fn get_random_lls(data: Vec<BlimpPyItem>, seed: u64) -> Vec<BlimpPyItem> {
        let mut rng = Pcg32::seed_from_u64(seed);
        data.into_iter()
            .map(|item| BlimpPyItem {
                ll_sentence_bad: rng.gen_range(0.0..(1.0 as f64)),
                ll_sentence_good: rng.gen_range(0.0..(1.0 as f64)),
                ..item
            })
            .collect()
    }

    pub(crate) fn register_eval_utils_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let eval_mod = PyModule::new_bound(parent.py(), "eval_utils")?;
        eval_mod.add_function(wrap_pyfunction!(get_ngrams_lls, &eval_mod)?)?;
        eval_mod.add_function(wrap_pyfunction!(get_bootstrap_lls, &eval_mod)?)?;
        eval_mod.add_function(wrap_pyfunction!(get_softmax_lls, &eval_mod)?)?;
        eval_mod.add_function(wrap_pyfunction!(get_random_lls, &eval_mod)?)?;

        crate::python::add_submodule(parent, &eval_mod)?;
        Ok(())
    }
}
