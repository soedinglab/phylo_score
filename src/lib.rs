#![allow(non_snake_case)]

pub mod felsenstein;
pub mod io;
pub mod models;

use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct Result {
    pub ll: f64,
    pub reg: f64,
    pub mi: f64,
}

pub fn optimize_model(
    model: &mut impl models::Model,
    f: felsenstein::FelsensteinOp,
    max_iterations: usize,
    learning_rate: f64,
    relative_stopping_threshold: f64,
) -> Result {
    let mut opt = candle_nn::optim::AdamW::new_lr(model.variables(), learning_rate).unwrap();
    let parameter = candle_nn::optim::ParamsAdamW {
        lr: learning_rate,
        weight_decay: 0.0,
        ..Default::default()
    };
    opt.set_params(parameter);
    const PATIENCE: usize = 5;
    let mut best_loss = f64::INFINITY;
    let mut no_improve_iterations = 0_usize;

    for _ in 0..max_iterations {
        let loss = model.loss(f.clone());
        let loss_value = loss.to_scalar().unwrap();
        candle_nn::optim::Optimizer::backward_step(&mut opt, &loss).unwrap();

        if best_loss.is_infinite() {
            best_loss = loss_value;
            continue;
        }

        let relative_improvement = (best_loss - loss_value) / best_loss.abs().max(1e-12);
        if relative_improvement > relative_stopping_threshold {
            best_loss = loss_value;
            no_improve_iterations = 0;
        } else {
            no_improve_iterations += 1;
            if no_improve_iterations >= PATIENCE {
                break;
            }
        }
    }

    let ll = model.ll(f).to_scalar().unwrap();
    let reg = model.reg().to_scalar().unwrap();
    let mi = model.mi().to_scalar().unwrap();
    Result { ll, reg, mi }
}

#[pyfunction]
#[pyo3(name = "optimize_models")]
fn optimize_models_py<'py>(
    py: Python<'py>,
    newick: String,
    fasta_file: String,
    reg_strength: f64,
    mi_strength: f64,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let newick_tree = std::fs::read_to_string(&newick).expect("Could not read newick file");
    let sequences = crate::io::read_alignment(&std::path::Path::new(&fasta_file));

    let L = sequences.values().next().unwrap().len();

    let mut pairs = vec![];
    for i in 0..L {
        for j in i + 1..L {
            pairs.push((i, j));
        }
    }

    let (felsenstein, leaf_pl) =
        crate::io::process_newick_alignment(&newick_tree, &sequences, pairs.iter().copied());

    let felsenstein = Box::leak(Box::new(felsenstein));

    let leaf_pl = leaf_pl
        .into_iter()
        .map(|mut col| {
            col.resize(
                felsenstein.num_nodes(),
                phylo_grad::nalgebra::SVector::<f64, 16>::zeros(),
            );
            std::sync::Arc::new(std::sync::Mutex::new(col))
        })
        .collect::<Vec<_>>();

    let results = leaf_pl
        .into_par_iter()
        .map(|pair| {
            let mut model = crate::models::IndModel::new(reg_strength);
            let ind = crate::optimize_model(
                &mut model,
                crate::felsenstein::FelsensteinOp::new(felsenstein, pair.clone()),
                2000,
                0.01,
                1e-6,
            );
            let mut pair_model = crate::models::PairModel::new(mi_strength, &model);
            let pair = crate::optimize_model(
                &mut pair_model,
                crate::felsenstein::FelsensteinOp::new(felsenstein, pair),
                2000,
                0.005,
                1e-6,
            );
            (ind, pair)
        })
        .collect::<Vec<_>>();

    let ind_ll = results.iter().map(|(ind, _)| ind.ll).collect::<Vec<_>>();
    let pair_ll = results.iter().map(|(_, pair)| pair.ll).collect::<Vec<_>>();
    let ind_reg = results.iter().map(|(ind, _)| ind.reg).collect::<Vec<_>>();
    let pair_reg = results.iter().map(|(_, pair)| pair.reg).collect::<Vec<_>>();

    (
        PyArray1::from_vec(py, ind_ll),
        PyArray1::from_vec(py, pair_ll),
        PyArray1::from_vec(py, ind_reg),
        PyArray1::from_vec(py, pair_reg),
    )
}

#[pymodule]
#[pyo3(name = "_phylo_score")]
fn phylo_score_mod<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_models_py, m)?)?;
    Ok(())
}
