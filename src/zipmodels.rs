use crate::utils::{LanguageModel, NonNanF64};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use std::cmp;
use std::f64::consts::E;
use std::io::Write;
use std::time::Instant;

#[inline(always)]
fn get_compressed_size(bytes: &[u8]) -> usize {
    let mut writer = ZlibEncoder::new(Vec::new(), Compression::best());
    writer
        .write_all(bytes)
        .expect("Writing to a vector won't cause an IO error");
    writer
        .finish()
        .expect("Writing to a vector won't cause an IO error")
        .len()
}

#[inline(always)]
fn get_concat_size(first: &[u8], second: &[u8]) -> usize {
    let mut writer = ZlibEncoder::new(Vec::new(), Compression::best());
    writer
        .write_all(first)
        .expect("Writing to a vector won't cause an IO error");
    writer
        .write_all(second)
        .expect("Writing to a vector won't cause an IO error");
    writer
        .finish()
        .expect("Writing to a vector won't cause an IO error")
        .len()
}

/// The ncd formla taken verbatim from the Jiang et al paper.
///
/// This takes the compressed lengths of the first, the second, and the combined pieces of text
/// and returns the ncd value.
#[inline(always)]
fn ncd(xlen: usize, ylen: usize, clen: usize) -> NonNanF64 {
    NonNanF64::new((clen - cmp::min(xlen, ylen)) as f64 / cmp::max(xlen, ylen) as f64)
        .expect("The ncd does not return NaN values")
}

fn get_distances<'a, T>(items: T, item: &(usize, &[u8])) -> Vec<NonNanF64>
where
    T: IntoIterator<Item = &'a (usize, Vec<u8>)>,
{
    items
        .into_iter()
        .map(|(xlen, bytes)| ncd(*xlen, item.0, get_concat_size(bytes, &item.1)))
        .collect()
}

fn cashe_compressed(items: Vec<Vec<u8>>) -> Vec<(usize, Vec<u8>)> {
    items
        .into_par_iter()
        .map(|item| (get_compressed_size(&item), item))
        .collect()
}

pub type LocFunc = fn(Vec<NonNanF64>) -> NonNanF64;

// A "language model" that calculates the likelyhood making use of a bootstap statistical test
// Technically, it doesn't calcualte log likelyhoods in the formal sence (hence the quotes),
// Instead, it calcualtes the chance that a piece of text is from the same population as the
// training data, by calculating the ncd between the training data, and comparing it to ncd from
// the items in the training set to the new pice of text.
// This means the log likelyhoods cannot be directly compared to those from any other model.
#[derive(Debug, Clone)]
pub struct BootstrapZipModel {
    items: Vec<(usize, Vec<u8>)>,
    tstar: Vec<NonNanF64>,
    t: LocFunc,
}

impl BootstrapZipModel {
    /// Create a new bootstrap model. With items, b, n, t, and some rng.
    ///
    /// b is the size of Tstar (in the traditional sense of a statistical bootstrap test)
    /// n is the amount of items to pull from every xstar.
    ///     Ideally, this would be `items.len()`, but this leads to long training times.
    /// t is the location function to use for the bootratap test.
    ///     If `n != items.len()`, this location metric must be equivilant on randome samples of the data.
    /// rng is the Rng to use when doing the random sampling of the bootstrap test.
    pub fn new<R>(items: Vec<Vec<u8>>, b: usize, n: usize, t: LocFunc, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let before = Instant::now();
        let items = cashe_compressed(items);
        println!("Compressed individual items in: {:.2?}", before.elapsed());

        let before = Instant::now();
        let mut tstar = items
            .choose_multiple(rng, b)
            .map(|(s, i)| (s, i, items.choose_multiple(rng, n)))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(s, i, xs)| t(get_distances(xs, &(*s, i))))
            .collect::<Vec<NonNanF64>>();
        tstar.sort_unstable();
        println!("Calculated TStar in: {:.2?}", before.elapsed());

        Self { items, tstar, t }
    }

    /// Calcualte the log likelyhood of a sentence.
    ///
    /// Keep in mind this is not a likelyhood in the traditional sense,
    /// Instead, it calcualtes the chance that a piece of text is from the same population as the
    /// training data, by calculating the ncd between the training data, and comparing it to ncd
    /// from the items in the training set to the new pice of text.
    pub fn get_log_likelyhood(&self, sentence: &[u8]) -> f64 {
        let before = Instant::now();
        let distances = get_distances(&self.items, &(get_compressed_size(sentence), sentence));
        let count_left = match self.tstar.binary_search(&(self.t)(distances)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };
        println!("Calcualted Log Likelyhood in: {:.2?}", before.elapsed());
        (count_left as f64 / self.tstar.len() as f64).log10()
    }
}

impl LanguageModel for BootstrapZipModel {
    fn get_log_likelyhood_sentence(&self, sentence: &str) -> f64 {
        self.get_log_likelyhood(sentence.as_bytes())
    }
}

#[derive(Debug, Clone)]
pub struct SoftmaxZipModel {
    pub items: Vec<(usize, Vec<u8>)>,
    pub softmax: f64,
    pub t: LocFunc,
}

impl SoftmaxZipModel {
    /// Create a new softmax model. With items, sample_size, t, and some rng.
    ///
    /// the sample size is the amount of items to pick to aproximate the denominator of the
    ///     softmax function. The quality of this aproximation does not matter when only comparing
    ///     the results of this model to itself, so chosing a value as low as 10 could be fine.
    ///     Keep in mind that this calculation can take up to a second per item to use.
    /// the t is some location metric to use to agrevate the ncd-distrances.
    /// the rng is the rng that is used to pick `sample_size` items from `items`.
    pub fn new<R>(items: Vec<Vec<u8>>, sample_size: usize, t: LocFunc, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let before = Instant::now();
        let items = cashe_compressed(items);
        println!("Compressed individual items in: {:.2?}", before.elapsed());

        // This is to calcualte the denominator of the softmax function,
        // which is pre-computed as it stays the same for each log likelyhood.
        // It has to be aproximated, as calcualting this score for every value in the training
        // set is extreemly expencive (taking months on my machine)
        // This approximation does not matter alot, tough: it is a constant devider, so
        // it doesn't affect anything when comparing the likelyhoods with eachther,
        // which, for now, is all that we do.
        let before = Instant::now();
        let sample_dists_sum = items
            .choose_multiple(rng, sample_size)
            .map(|(s, i)| (s, i, items.iter().filter(|(_, item)| **item != **i)))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(s, i, xs)| t(get_distances(xs, &(*s, i))).value())
            .map(|val| E.powf(val))
            .sum::<f64>();
        let softmax = sample_dists_sum * (items.len() as f64 / sample_size as f64);
        println!("Calculated Softmax den approx in: {:.2?}", before.elapsed());

        Self { items, softmax, t }
    }

    pub fn get_log_likelyhood(&self, sentence: &[u8]) -> f64 {
        let before = Instant::now();
        let distances = get_distances(&self.items, &(get_compressed_size(sentence), sentence));
        println!("Calcualted Log Likelyhood in: {:.2?}", before.elapsed());
        (E.powf((self.t)(distances).value()) / self.softmax).log10()
    }
}

impl LanguageModel for SoftmaxZipModel {
    fn get_log_likelyhood_sentence(&self, sentence: &str) -> f64 {
        self.get_log_likelyhood(sentence.as_bytes())
    }
}

pub mod python {
    #![allow(unused)]
    use super::*;
    use crate::utils::python::*;
    use crate::utils::Corpus;
    use pyo3::{exceptions::PyValueError, prelude::*};
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    fn quant_from_index(t_type: usize) -> PyResult<LocFunc> {
        match t_type {
            0 => Ok(|xs| NonNanF64::mean(xs)),
            1 => Ok(|xs| NonNanF64::quantile(xs, 0.25)),
            2 => Ok(|xs| NonNanF64::quantile(xs, 0.50)),
            3 => Ok(|xs| NonNanF64::quantile(xs, 0.75)),
            _ => Err(PyValueError::new_err("{t} is not a valid function type")),
        }
    }

    create_pycorpus!(PyCorpusBytes, Vec<Vec<u8>>);

    #[pyfunction]
    pub fn bytes_pycorpus(sentences: &PyCorpusSentences) -> PyCorpusBytes {
        let items = sentences
            .items()
            .iter()
            .map(|string| string.as_bytes().to_vec())
            .collect();
        PyCorpusBytes::new(items)
    }

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct PyBootstrapZipModel {
        pub inner: BootstrapZipModel,
    }

    #[pymethods]
    impl PyBootstrapZipModel {
        #[new]
        pub fn new(
            items: PyCorpusBytes,
            b: usize,
            n: usize,
            t_type: usize,
            seed: u64,
        ) -> PyResult<Self> {
            let mut rng = Pcg32::seed_from_u64(seed);
            let t = quant_from_index(t_type)?;

            let inner = BootstrapZipModel::new(items.items(), b, n, t, &mut rng);
            Ok(Self { inner })
        }
        pub fn get_log_likelyhood(&self, items: Vec<u8>) -> f64 {
            self.inner.get_log_likelyhood(&items)
        }
    }

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct PySoftmaxZipModel {
        pub inner: SoftmaxZipModel,
    }

    #[pymethods]
    impl PySoftmaxZipModel {
        #[new]
        pub fn new(
            items: PyCorpusBytes,
            sample_size: usize,
            t_type: usize,
            seed: u64,
        ) -> PyResult<Self> {
            let mut rng = Pcg32::seed_from_u64(seed);
            let t = quant_from_index(t_type)?;

            let inner = SoftmaxZipModel::new(items.items(), sample_size, t, &mut rng);
            Ok(Self { inner })
        }
        pub fn get_log_likelyhood(&self, items: Vec<u8>) -> f64 {
            self.inner.get_log_likelyhood(&items)
        }
    }

    pub(crate) fn register_zipmodels_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let zipmodels_mod = PyModule::new_bound(parent.py(), "zipmodels")?;
        zipmodels_mod.add_class::<PyCorpusBytes>()?;
        zipmodels_mod.add_function(wrap_pyfunction!(bytes_pycorpus, &zipmodels_mod)?)?;
        zipmodels_mod.add_class::<PyBootstrapZipModel>()?;
        zipmodels_mod.add_class::<PySoftmaxZipModel>()?;

        crate::python::add_submodule(parent, &zipmodels_mod)?;
        Ok(())
    }
}
