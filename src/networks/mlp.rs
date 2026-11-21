use tch::{Device, nn};

pub struct MLP {
    pub model: nn::Sequential,
    pub var_store: nn::VarStore,
}

impl MLP {
    pub fn new(device: Device, input_dim: usize, output_dim: usize) -> Self {
        let var_store = nn::VarStore::new(device);
        let model = nn::seq()
            .add(nn::linear(
                &var_store.root() / "layer1",
                input_dim as i64,
                64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &var_store.root() / "output",
                64,
                output_dim as i64,
                Default::default(),
            ));
        MLP { model, var_store }
    }
}
