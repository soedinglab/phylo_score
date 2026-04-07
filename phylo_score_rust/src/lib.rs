#![allow(non_snake_case)]

pub mod felsenstein;
pub mod models;
pub mod io;

pub struct Result {
    pub ll: f64,
    pub reg: f64,
    pub mi: f64,
}

pub fn optimize_model(
    model: &mut impl models::Model,
    f : felsenstein::FelsensteinOp,
    max_iterations: usize,
    learning_rate: f64,
    relative_stopping_threshold: f64
) -> Result {
    let mut opt = candle_nn::optim::AdamW::new_lr(
        model.variables(),
        learning_rate,
    ).unwrap();
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
