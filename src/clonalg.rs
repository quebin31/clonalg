use crate::utils::ToDisplayPath;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::ops::{Deref, DerefMut};

pub trait Fitness {
    fn call(&self, path: &[usize], distances: &Array2<f64>) -> f64;
}

impl<F> Fitness for F
where
    F: Fn(&[usize], &Array2<f64>) -> f64,
{
    fn call(&self, path: &[usize], distances: &Array2<f64>) -> f64 {
        (self)(path, distances)
    }
}

#[derive(Debug, Clone)]
pub struct Path(Vec<usize>);

impl Deref for Path {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Path {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Path {
    pub fn mutate(&mut self) -> (usize, usize) {
        let mut rng = thread_rng();

        let i = rng.gen_range(0, self.len());
        let j = {
            let mut rand = rng.gen_range(0, self.len());
            while rand == i {
                rand = rng.gen_range(0, self.len());
            }

            rand
        };

        self.swap(i, j);
        (i, j)
    }
}

/// Function to initialize a random population
pub fn random_population(size: usize, no_cities: usize) -> Vec<Path> {
    let mut rng = thread_rng();
    let mut vec: Vec<_> = (0..no_cities).collect();

    (0..size)
        .map(|_| {
            vec.shuffle(&mut rng);
            Path(vec.clone())
        })
        .collect()
}

pub fn select_subset<F>(
    size: usize,
    population: &[Path],
    fitness: F,
    distances: &Array2<f64>,
) -> Vec<Path>
where
    F: Fitness,
{
    let mut parents = population.to_owned();
    parents.sort_by(|a, b| {
        let fit_a = fitness.call(a, distances);
        let fit_b = fitness.call(b, distances);

        fit_a.partial_cmp(&fit_b).unwrap()
    });

    parents.drain(0..size).collect()
}

/// Function to create a set of clones from the population
pub fn clonal_expansion<F>(parents: &[Path], fitness: F, distances: &Array2<f64>) -> Vec<Path>
where
    F: Fitness,
{
    let mut sorted = parents.to_owned();
    sorted.sort_by(|a, b| {
        let fit_a = fitness.call(a, distances);
        let fit_b = fitness.call(b, distances);

        fit_a.partial_cmp(&fit_b).unwrap()
    });

    let mut clones = Vec::new();
    let len = sorted.len();
    for (i, path) in sorted.into_iter().enumerate() {
        let no_clones = len - i;
        (0..no_clones).for_each(|_| clones.push(path.clone()));
    }

    clones
}

pub fn hypermutation_inplace<F>(
    parent_size: usize,
    clones: &mut [Path],
    fitness: F,
    distances: &Array2<f64>,
) where
    F: Fitness,
{
    let mut mutated = 0;
    let mut no_mutations = 1;

    for (i, path) in clones.iter_mut().enumerate() {
        if mutated == parent_size - no_mutations + 1 {
            mutated = 0;
            no_mutations += 1;
        }

        print!("{}) {} ", i + 1, path.to_display_path().unwrap());

        for _ in 0..no_mutations {
            let (i, j) = path.mutate();
            print!("({}, {})", i, j);
        }

        println!(
            " {} fitness: {}",
            path.to_display_path().unwrap(),
            fitness.call(&path, distances),
        );

        mutated += 1;
    }
}

pub fn combine_and_replace<F>(
    population: &mut [Path],
    mut selected: Vec<Path>,
    mut random: Vec<Path>,
    fitness: F,
    distances: &Array2<f64>,
) where
    F: Fitness,
{
    population.sort_by(|a, b| {
        let fit_a = fitness.call(a, distances);
        let fit_b = fitness.call(b, distances);

        fit_a.partial_cmp(&fit_b).unwrap()
    });

    let mut combined = Vec::with_capacity(selected.len() + random.len());
    combined.append(&mut selected);
    combined.append(&mut random);

    combined.sort_by(|a, b| {
        let fit_a = fitness.call(a, distances);
        let fit_b = fitness.call(b, distances);

        fit_a.partial_cmp(&fit_b).unwrap()
    });

    for (worst, better) in population.iter_mut().rev().zip(combined) {
        let fit_worst = fitness.call(worst, distances);
        let fit_better = fitness.call(&better, distances);

        if fit_worst < fit_better {
            break;
        } else {
            *worst = better;
        }
    }
}
