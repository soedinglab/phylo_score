#![allow(non_snake_case)]

use std::{path::Path, sync::Arc};

use candle_core::Tensor;
use _phylo_score::felsenstein::FelsensteinOp;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let newick = args[1].clone();
    let newick_tree = std::fs::read_to_string(&newick).expect("Could not read newick file");
    let fasta_file = args[2].clone();
    let reg_strength = args[3].parse::<f64>().expect("Could not parse regularization strength");
    let mi_strength = args[4].parse::<f64>().expect("Could not parse mutual information regularization strength");
    let output_file = args[5].clone();
    let sequences = _phylo_score::io::read_alignment(&Path::new(&fasta_file));

    let L = sequences.values().next().unwrap().len();

    let mut pairs = vec![];
    for i in 0..L {
        for j in i + 1..L {
            pairs.push((i, j));
        }
    }

    let (felsenstein, leaf_pl) =
        _phylo_score::io::process_newick_alignment(&newick_tree, &sequences, pairs.iter().copied());

    let felsenstein = Box::leak(Box::new(felsenstein));

    let leaf_pl = leaf_pl
        .into_iter()
        .map(|mut col| {
            col.resize(
                felsenstein.num_nodes(),
                phylo_grad::nalgebra::SVector::<f64, 16>::zeros(),
            );
            Arc::new(std::sync::Mutex::new(col))
        })
        .collect::<Vec<_>>();

    let results = leaf_pl
        .into_par_iter()
        .map(|pair| {
            let mut model = _phylo_score::models::IndModel::new(reg_strength);
            let ind_ll = _phylo_score::optimize_model(
                &mut model,
                FelsensteinOp::new(felsenstein, pair.clone()),
                2000,
                0.01,
                1e-6,
            );
            let mut pair_model = _phylo_score::models::PairModel::new(mi_strength, &model);
            let pair_ll = _phylo_score::optimize_model(
                &mut pair_model,
                FelsensteinOp::new(felsenstein, pair),
                2000,
                0.005,
                1e-6,
            );
            (ind_ll, pair_ll)
        })
        .collect::<Vec<_>>();

    let log_odds = results.iter().map(|(ind, pair)| pair.ll - ind.ll).collect::<Vec<_>>();
    
    let log_odds = Tensor::from_vec(log_odds, &[pairs.len()], &candle_core::Device::Cpu).unwrap();

    let mi = results.iter().map(|(_, pair)| pair.mi).collect::<Vec<_>>();
    let mi = Tensor::from_vec(mi, &[pairs.len()], &candle_core::Device::Cpu).unwrap();

    let ind_ll = results.iter().map(|(ind, _)| ind.ll).collect::<Vec<_>>();
    let ind_ll = Tensor::from_vec(ind_ll, &[pairs.len()], &candle_core::Device::Cpu).unwrap();

    let pair_ll = results.iter().map(|(_, pair)| pair.ll).collect::<Vec<_>>();
    let pair_ll = Tensor::from_vec(pair_ll, &[pairs.len()], &candle_core::Device::Cpu).unwrap();

    let ind_reg = results.iter().map(|(ind, _)| ind.reg).collect::<Vec<_>>();
    let ind_reg = Tensor::from_vec(ind_reg, &[pairs.len()], &candle_core::Device::Cpu).unwrap();

    let pair_reg = results.iter().map(|(_, pair)| pair.reg).collect::<Vec<_>>();
    let pair_reg = Tensor::from_vec(pair_reg, &[pairs.len()], &candle_core::Device::Cpu).unwrap();

    Tensor::write_npz(&[("log_odds".to_string(), log_odds), ("mi".to_string(), mi), ("ind_ll".to_string(), ind_ll), ("pair_ll".to_string(), pair_ll), ("ind_reg".to_string(), ind_reg), ("pair_reg".to_string(), pair_reg)], &output_file).unwrap();
}
