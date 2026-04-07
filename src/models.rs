use std::vec;

use candle_core::{Tensor, Var};

use crate::felsenstein::FelsensteinOp;

pub trait Model {
    fn get_S(&self) -> Tensor;
    fn get_log_pi(&self) -> Tensor;
    fn reg(&self) -> Tensor;
    fn variables(&self) -> Vec<Var>;
    fn get_sqrt_pi(&self) -> Tensor {
        (self.get_log_pi() / 2.0).unwrap().exp().unwrap()
    }
    fn loss(&self, f: FelsensteinOp) -> Tensor {
        let reg = self.reg();
        return (self.ll(f).neg().unwrap() + reg).unwrap();
    }
    fn ll(&self, f: FelsensteinOp) -> Tensor {
        let S = self.get_S();
        let sqrt_pi = self.get_sqrt_pi();
        return S.apply_op2(&sqrt_pi, f).unwrap();
    }
    fn mi(&self) -> Tensor;
}

pub struct IndModel {
    log_rate1: Var,
    log_rate2: Var,

    log_pi1: Var,
    log_pi2: Var,

    reg_strength: f64,

    s_indices: Tensor,
}

pub struct PairModel {
    log_rate1: Var,
    log_rate2: Var,
    log_pi: Var,

    reg_strength: f64,
    mi_reg_strength: f64,

    s_indices: Tensor,
}

impl IndModel {
    pub fn new(reg_strength: f64) -> Self {
        let device = candle_core::Device::Cpu;
        let dtype = candle_core::DType::F64;
        Self {
            log_rate1: Var::zeros(&[4, 4], dtype, &device).unwrap(),
            log_rate2: Var::zeros(&[4, 4], dtype, &device).unwrap(),
            log_pi1: Var::zeros(&[4], dtype, &device).unwrap(),
            log_pi2: Var::zeros(&[4], dtype, &device).unwrap(),
            reg_strength,
            s_indices: get_combine_S_indicies(),
        }
    }
}

impl PairModel {
    pub fn new(mi_reg_strength: f64, ind_model: &IndModel) -> Self {
        Self {
            log_rate1: Var::from_tensor(&ind_model.log_rate1.copy().unwrap()).unwrap(),
            log_rate2: Var::from_tensor(&ind_model.log_rate2.copy().unwrap()).unwrap(),
            log_pi: Var::from_tensor(&ind_model.get_log_pi().copy().unwrap()).unwrap(),
            reg_strength: ind_model.reg_strength,
            mi_reg_strength,
            s_indices: get_combine_S_indicies(),
        }
    }
}

fn get_combine_S_indicies() -> Tensor {
    let mut indices = vec![0_i64; 16 * 16];
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let position = (a * 4 + b) * 16 + (c * 4 + d);
                    if a == c && b != d {
                        // Rate1
                        indices[position] = (4 * b + d) as i64;
                    } else if a != c && b == d {
                        indices[position] = (16 + 4 * a + c) as i64;
                    } else {
                        // Zero
                        indices[position] = 32;
                    }
                }
            }
        }
    }

    return Tensor::from_vec(indices, &[16 * 16], &candle_core::Device::Cpu).unwrap();
}

impl Model for IndModel {
    fn get_S(&self) -> Tensor {
        let rate1 = (self.log_rate1.as_tensor() + self.log_rate1.transpose(0, 1).unwrap())
            .unwrap()
            .exp()
            .unwrap();
        let rate2 = (self.log_rate2.as_tensor() + self.log_rate2.transpose(0, 1).unwrap())
            .unwrap()
            .exp()
            .unwrap();
        let all = [
            rate1.flatten_all().unwrap(),
            rate2.flatten_all().unwrap(),
            Tensor::zeros(&[1], candle_core::DType::F64, &candle_core::Device::Cpu).unwrap(),
        ];
        let all = Tensor::cat(&all, 0).unwrap();
        return all
            .gather(&self.s_indices, 0)
            .unwrap()
            .reshape(&[16, 16])
            .unwrap();
    }

    fn get_log_pi(&self) -> Tensor {
        let pi1: Tensor = candle_nn::ops::log_softmax(self.log_pi1.as_tensor(), 0).unwrap();
        let pi2: Tensor = candle_nn::ops::log_softmax(self.log_pi2.as_tensor(), 0).unwrap();

        let pi = pi1
            .unsqueeze(1)
            .unwrap()
            .broadcast_add(&pi2.unsqueeze(0).unwrap())
            .unwrap();

        return pi.reshape(&[16]).unwrap();
    }

    fn reg(&self) -> Tensor {
        let rate1 = (self.log_rate1.as_tensor() + self.log_rate1.transpose(0, 1).unwrap()).unwrap();
        let rate2 = (self.log_rate2.as_tensor() + self.log_rate2.transpose(0, 1).unwrap()).unwrap();

        let mask = (Tensor::ones_like(&rate1).unwrap()
            - Tensor::eye(4, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
        .unwrap();

        let rate1 = (rate1 * &mask).unwrap();
        let rate2 = (rate2 * &mask).unwrap();

        let reg_rate1 = (&rate1 * &rate1)
            .unwrap()
            .sum_all()
            .unwrap()
            .broadcast_div(&Tensor::from_slice(&[2.0], &[], &candle_core::Device::Cpu).unwrap())
            .unwrap();
        let reg_rate2 = (&rate2 * &rate2)
            .unwrap()
            .sum_all()
            .unwrap()
            .broadcast_div(&Tensor::from_slice(&[2.0], &[], &candle_core::Device::Cpu).unwrap())
            .unwrap();

        let log_pi1 = self
            .log_pi1
            .as_tensor()
            .broadcast_sub(&self.log_pi1.mean(0).unwrap())
            .unwrap();
        let log_pi2 = self
            .log_pi2
            .as_tensor()
            .broadcast_sub(&self.log_pi2.mean(0).unwrap())
            .unwrap();

        let reg_pi1 = (&log_pi1 * &log_pi1).unwrap().sum_all().unwrap();
        let reg_pi2 = (&log_pi2 * &log_pi2).unwrap().sum_all().unwrap();

        return ((reg_rate1 + reg_rate2 + reg_pi1 + reg_pi2).unwrap() * self.reg_strength).unwrap();
    }

    fn variables(&self) -> Vec<Var> {
        vec![
            self.log_rate1.clone(),
            self.log_rate2.clone(),
            self.log_pi1.clone(),
            self.log_pi2.clone(),
        ]
    }

    fn mi(&self) -> Tensor {
        return Tensor::zeros(&[], candle_core::DType::F64, &candle_core::Device::Cpu).unwrap();
    }
}

impl Model for PairModel {
    fn get_S(&self) -> Tensor {
        let rate1 = (self.log_rate1.as_tensor() + self.log_rate1.transpose(0, 1).unwrap())
            .unwrap()
            .exp()
            .unwrap();
        let rate2 = (self.log_rate2.as_tensor() + self.log_rate2.transpose(0, 1).unwrap())
            .unwrap()
            .exp()
            .unwrap();
        let all = [
            rate1.flatten_all().unwrap(),
            rate2.flatten_all().unwrap(),
            Tensor::zeros(&[1], candle_core::DType::F64, &candle_core::Device::Cpu).unwrap(),
        ];
        let all = Tensor::cat(&all, 0).unwrap();
        return all
            .gather(&self.s_indices, 0)
            .unwrap()
            .reshape(&[16, 16])
            .unwrap();
    }

    fn get_log_pi(&self) -> Tensor {
        return candle_nn::ops::log_softmax(self.log_pi.as_tensor(), 0).unwrap();
    }

    fn mi(&self) -> Tensor {
        let log_pi = self.get_log_pi().reshape(&[4, 4]).unwrap();

        let log_pi1 = log_pi.log_sum_exp(1).unwrap();
        let log_pi2 = log_pi.log_sum_exp(0).unwrap();

        let log_independent = log_pi1
            .unsqueeze(1)
            .unwrap()
            .broadcast_add(&log_pi2.unsqueeze(0).unwrap())
            .unwrap();

        let log_ratio = (&log_pi - log_independent).unwrap();

        let mi = (log_ratio * log_pi.exp()).unwrap().sum_all().unwrap();
        return mi;
    }

    fn reg(&self) -> Tensor {
        let rate1 = (self.log_rate1.as_tensor() + self.log_rate1.transpose(0, 1).unwrap()).unwrap();
        let rate2 = (self.log_rate2.as_tensor() + self.log_rate2.transpose(0, 1).unwrap()).unwrap();

        let mask = (Tensor::ones_like(&rate1).unwrap()
            - Tensor::eye(4, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
        .unwrap();

        let rate1 = (rate1 * &mask).unwrap();
        let rate2 = (rate2 * &mask).unwrap();

        let reg_rate1 = (&rate1 * &rate1)
            .unwrap()
            .sum_all()
            .unwrap()
            .broadcast_div(&Tensor::from_slice(&[2.0], &[], &candle_core::Device::Cpu).unwrap())
            .unwrap();
        let reg_rate2 = (&rate2 * &rate2)
            .unwrap()
            .sum_all()
            .unwrap()
            .broadcast_div(&Tensor::from_slice(&[2.0], &[], &candle_core::Device::Cpu).unwrap())
            .unwrap();

        let log_pi = self.get_log_pi().reshape(&[4, 4]).unwrap();

        let log_pi1 = log_pi.log_sum_exp(1).unwrap();
        let log_pi2 = log_pi.log_sum_exp(0).unwrap();

        let log_independent = log_pi1
            .unsqueeze(1)
            .unwrap()
            .broadcast_add(&log_pi2.unsqueeze(0).unwrap())
            .unwrap();

        let log_pi1 = log_pi1.broadcast_sub(&log_pi1.mean(0).unwrap()).unwrap();
        let log_pi2 = log_pi2.broadcast_sub(&log_pi2.mean(0).unwrap()).unwrap();


        let log_ratio = (&log_pi - log_independent).unwrap();

        let mi = (log_ratio * log_pi.exp()).unwrap().sum_all().unwrap();

        let reg_pi1 = (&log_pi1 * &log_pi1).unwrap().sum_all().unwrap();
        let reg_pi2 = (&log_pi2 * &log_pi2).unwrap().sum_all().unwrap();

        return (((reg_rate1 + reg_rate2 + reg_pi1 + reg_pi2).unwrap() * self.reg_strength)
            .unwrap()
            + (mi * self.mi_reg_strength).unwrap())
        .unwrap();
    }

    fn variables(&self) -> Vec<Var> {
        vec![
            self.log_rate1.clone(),
            self.log_rate2.clone(),
            self.log_pi.clone(),
        ]
    }
}
