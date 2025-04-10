use tch::{Device, Tensor};

pub trait ToTensor {
    fn to_tensor(&self) -> Tensor;
}

impl ToTensor for [f32; 4] {
    fn to_tensor(&self) -> Tensor {
        Tensor::from_slice(self).unsqueeze(0)
    }
}
