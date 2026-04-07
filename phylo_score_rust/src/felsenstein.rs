use std::{sync::{Arc, Mutex}, vec};

use candle_core::CpuStorage;

pub struct FelsensteinOp {
    felsenstein : &'static phylo_grad::FelsensteinTree<16>,
    result_storage: Mutex<Option<phylo_grad::SingleSideResult<f64, 16>>>,
    partial_likelihoods : Arc<Mutex<Vec<phylo_grad::nalgebra::SVector<f64, 16>>>>,
}


fn tensor2nalgebra2d(s: &candle_core::CpuStorage, l: &candle_core::Layout) -> phylo_grad::nalgebra::SMatrix<f64, 16, 16> {
    let dims = l.shape().dims();
    let stride = l.stride();
    assert!(dims.len() == 2);
    match s {
        CpuStorage::F64(data) => {
            let offset = l.start_offset();
            let mut mat = phylo_grad::nalgebra::SMatrix::<f64, 16, 16>::zeros();
            for i in 0..16 {
                for j in 0..16 {
                    mat[(i, j)] = data[offset + i * stride[0] + j * stride[1]];
                }
            }
            mat
        }
        _ => panic!("Expected f64 storage."),
    }
}

fn _tensor2vec_nalebgra1d(s: &candle_core::CpuStorage, l: &candle_core::Layout) -> Vec<phylo_grad::nalgebra::SVector<f64, 16>> {
    let dims = l.shape().dims();
    let stride = l.stride();
    assert!(dims.len() == 2);
    let batch_size = dims[0];
    let mut result = Vec::with_capacity(batch_size);
    match s {
        CpuStorage::F64(data) => {
            let mut offset = l.start_offset();
            for _ in 0..batch_size {
                let mut vec = phylo_grad::nalgebra::SVector::<f64, 16>::zeros();
                for i in 0..16 {
                    vec[i] = data[offset + i * stride[1]];
                }
                result.push(vec);
                offset += stride[0];
            }
        }
        _ => panic!("Expected f64 storage."),
    }
    result
}

fn tensor2nalgebra1d(s: &candle_core::CpuStorage, l: &candle_core::Layout) -> phylo_grad::nalgebra::SVector<f64, 16> {
    let dims = l.shape().dims();
    let stride = l.stride();
    assert!(dims.len() == 1);
   
    match s {
        CpuStorage::F64(data) => {
            let offset = l.start_offset();
            let mut vec = phylo_grad::nalgebra::SVector::<f64, 16>::zeros();
            for i in 0..16 {
                vec[i] = data[offset + i * stride[0]];
            }
            vec
        }
        _ => panic!("Expected f64 storage."),
    }
}

fn _tensor2vec(s: &candle_core::CpuStorage, l: &candle_core::Layout) -> Vec<f64> {
    let dims = l.shape().dims();
    let stride = l.stride();
    assert!(dims.len() == 1);
    let batch_size = dims[0];
    let mut result = Vec::with_capacity(batch_size);
    match s {
        CpuStorage::F64(data) => {
            let mut offset = l.start_offset();
            for _ in 0..batch_size {
                result.push(data[offset]);
                offset += stride[0];
            }
        }
        _ => panic!("Expected f64 storage."),
    }
    result
}

fn nalgebra2tensor1d(data: &phylo_grad::nalgebra::SVector<f64, 16>) -> candle_core::Tensor {
    let vec = data.as_slice().to_vec();
    candle_core::Tensor::from_vec(vec, &candle_core::Shape::from(&[16]), &candle_core::Device::Cpu).unwrap()
}

fn nalgebra2tensor2d(mats: &phylo_grad::nalgebra::SMatrix<f64, 16, 16>) -> candle_core::Tensor {
    let mut vec = Vec::with_capacity(16 * 16);
    for i in 0..16 {
        for j in 0..16 {
            vec.push(mats[(i, j)]);
        }
    }
    candle_core::Tensor::from_vec(vec, &candle_core::Shape::from(&[16, 16]), &candle_core::Device::Cpu).unwrap()
}

impl FelsensteinOp {
    pub fn new(felsenstein: &'static phylo_grad::FelsensteinTree<16>, pl : Arc<Mutex<Vec<phylo_grad::nalgebra::SVector<f64, 16>>>>) -> Self {
        Self {
            felsenstein,
            result_storage: Mutex::new(None),
            partial_likelihoods: pl,
        }
    }
}

impl Clone for FelsensteinOp {
    fn clone(&self) -> Self {
        Self {
            felsenstein: self.felsenstein,
            result_storage: Mutex::new(None),
            partial_likelihoods: Arc::clone(&self.partial_likelihoods),
        }
    }
}

impl candle_core::CustomOp2 for FelsensteinOp {
    fn name(&self) -> &'static str {
        "FelsensteinOp"
    }

    /// S, sqrt
    fn cpu_fwd(
            &self,
            s1: &candle_core::CpuStorage,
            l1: &candle_core::Layout,
            s2: &candle_core::CpuStorage,
            l2: &candle_core::Layout,
        ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        
        let s = tensor2nalgebra2d(s1, l1);
        let sqrt_pi = tensor2nalgebra1d(s2, l2);

        let mut result = self.felsenstein.calculate_gradients_single_side(s.as_view(), sqrt_pi.as_view(), self.partial_likelihoods.lock().unwrap().as_mut());

        let likelihood = std::mem::take(&mut result.log_likelihood);

        let mut storage = self.result_storage.lock().unwrap();
        *storage = Some(result);

        let shape = candle_core::Shape::from(&[]);
        let cpu_storage = candle_core::CpuStorage::F64(vec![likelihood]);
        return Ok((cpu_storage, shape))
    }

    fn bwd(
            &self,
            _arg1: &candle_core::Tensor,
            _arg2: &candle_core::Tensor,
            _res: &candle_core::Tensor,
            grad_res: &candle_core::Tensor,
        ) -> candle_core::Result<(Option<candle_core::Tensor>, Option<candle_core::Tensor>)> {
        let mut result = self.result_storage.lock().unwrap();
        

        let grad_s = std::mem::take(&mut result.as_mut().unwrap().grad_s);
        let grad_s_tensor = nalgebra2tensor2d(&grad_s);
        let grad_sqrt_pi = std::mem::take(&mut result.as_mut().unwrap().grad_sqrt_pi);
        let grad_sqrt_pi_tensor = nalgebra2tensor1d(&grad_sqrt_pi);


        Ok((
            Some(grad_res.broadcast_mul(&grad_s_tensor).unwrap()),
            Some(grad_res.broadcast_mul(&grad_sqrt_pi_tensor).unwrap()),
        ))

    }
}