use tch::Tensor;

pub trait ToTensor {
    fn to_tensor(&self) -> Tensor;
}

impl<const N: usize> ToTensor for [f32; N] {
    fn to_tensor(&self) -> Tensor {
        Tensor::from_slice(self).unsqueeze(0)
    }
}
