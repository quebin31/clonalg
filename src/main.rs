use anyhow::Error;
use clonalg::{
    clonal_expansion, combine_and_replace, hypermutation_inplace, random_population, select_subset,
};
use ndarray::Array2;
use utils::ToDisplayPath;

pub mod clonalg;
pub mod utils;

fn fitness(path: &[usize], distances: &Array2<f64>) -> f64 {
    path.windows(2)
        .fold(0.0, |acc, edge| acc + distances[[edge[0], edge[1]]])
}

macro_rules! print_population {
    ($pop:ident, $distances:ident) => {{
        for (i, path) in $pop.iter().enumerate() {
            println!(
                "{}) {} fitness: {}",
                i + 1,
                path.to_display_path()?,
                fitness(path, &$distances)
            );
        }
    }};
}

fn main() -> Result<(), Error> {
    let no_cities = 10;
    let distances: Vec<_> = [
        0, 1, 3, 23, 11, 5, 83, 21, 28, 45, //
        1, 0, 1, 18, 3, 41, 20, 61, 95, 58, //
        3, 1, 0, 1, 56, 21, 43, 17, 83, 16, //
        23, 18, 1, 0, 1, 46, 44, 45, 50, 11, //
        11, 3, 56, 1, 0, 1, 93, 38, 78, 41, //
        5, 41, 21, 46, 1, 0, 1, 90, 92, 97, //
        83, 20, 43, 44, 93, 1, 0, 1, 74, 29, //
        21, 61, 17, 45, 38, 90, 1, 0, 1, 28, //
        28, 95, 83, 50, 78, 92, 74, 1, 0, 1, //
        45, 58, 16, 11, 41, 97, 29, 28, 1, 0, //
    ]
    .iter()
    .map(|v| *v as f64)
    .collect();
    let distances = Array2::from_shape_vec((no_cities, no_cities), distances)?;

    let population_size = 7;
    let parents_size = 5;
    let selected_size = 5;
    let random_size = 2;
    let max_iters = 1;

    let mut population = random_population(population_size, no_cities);

    println!("*** Población P ***");
    print_population!(population, distances);

    for i in 0..max_iters {
        println!("\n**** Iteración {} ****", i + 1);

        let parents = select_subset(parents_size, &population, fitness, &distances);
        println!("*** Población F ***");
        print_population!(parents, distances);

        let mut clones = clonal_expansion(&parents, fitness, &distances);
        println!("\n*** Población P(clone) ***");
        print_population!(clones, distances);

        println!("\n*** Población P(hyper) ***");
        hypermutation_inplace(parents.len(), &mut clones, fitness, &distances);

        let selected = select_subset(selected_size, &clones, fitness, &distances);
        println!("\n*** Población S ***");
        print_population!(selected, distances);

        let random = random_population(random_size, no_cities);
        println!("\n*** Población R ***");
        print_population!(random, distances);

        combine_and_replace(&mut population, selected, random, fitness, &distances);
        println!("\n*** Población P ***");
        print_population!(population, distances);
    }

    Ok(())
}
