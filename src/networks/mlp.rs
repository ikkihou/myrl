use tch::{Device, nn};

pub struct MLP {
    pub model: nn::Sequential,
    pub var_store: nn::VarStore, // 👈 保存 VarStore 的所有权
}

impl MLP {
    pub fn new(vs: nn::VarStore, input_dim: usize, output_dim: usize) -> Self {
        let model = nn::seq()
            .add(nn::linear(
                &vs.root() / "layer1",
                input_dim.try_into().unwrap(),
                64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &vs.root() / "output",
                64,
                output_dim.try_into().unwrap(),
                Default::default(),
            ));
        MLP {
            model,
            var_store: vs,
        }
    }

    pub fn var_store(&self) -> &nn::VarStore {
        &self.var_store
    }

    pub fn device(&self) -> Device {
        self.var_store.device()
    }
}
