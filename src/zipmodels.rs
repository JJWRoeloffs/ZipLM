use crate::utils::NonNanF64;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use std::cmp;
use std::f64::consts::E;
use std::io::Write;
use std::marker::Sync;
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

#[derive(Debug)]
pub struct BootstrapZipModel<F: Fn(Vec<NonNanF64>) -> NonNanF64> {
    items: Vec<(usize, Vec<u8>)>,
    tstar: Vec<NonNanF64>,
    t: F,
}

impl<F: Fn(Vec<NonNanF64>) -> NonNanF64 + Sync> BootstrapZipModel<F> {
    pub fn new<R>(items: Vec<Vec<u8>>, b: usize, n: usize, t: F, rng: &mut R) -> Self
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

    pub fn get_log_likelyhood(&self, sentence: &[u8]) -> f64 {
        let before = Instant::now();
        let distances = get_distances(&self.items, &(get_compressed_size(sentence), sentence));
        let count_left = match self.tstar.binary_search(&(self.t)(distances)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };
        println!("Calcualted Log Likelyhood in: {:.2?}", before.elapsed());
        (1.0 - (count_left as f64 / self.tstar.len() as f64)).log10()
    }
}

#[derive(Debug)]
pub struct SoftmaxZipModel<F: Fn(Vec<NonNanF64>) -> NonNanF64 + Sync> {
    pub items: Vec<(usize, Vec<u8>)>,
    pub softmax: f64,
    pub t: F,
}

impl<F: Fn(Vec<NonNanF64>) -> NonNanF64 + Sync> SoftmaxZipModel<F> {
    pub fn new<R>(items: Vec<Vec<u8>>, sample_size: usize, t: F, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let before = Instant::now();
        let items = cashe_compressed(items);
        println!("Compressed individual items in: {:.2?}", before.elapsed());

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
