use crate::environment::Environment;
use crate::networks::mlp::MLP;
use crate::policy::Policy;
use crate::train::Train;
use crate::utils::ToTensor;

use tch::{
    Device, Tensor,
    nn::{self, OptimizerConfig},
};

struct TrajectoryBatch {
    observations: Tensor,
    actions: Tensor,
    rewards: Tensor,
    values: Tensor,
    log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
}

pub struct PPOAgent<E: Environment> {
    policy_net: MLP,
    value_net: MLP,
    pn_opt: nn::Optimizer,
    vn_opt: nn::Optimizer,
    pn_lr: f32,
    vn_lr: f32,
    gamma: f32,
    lambda: f32,
    clip_ratio: f32,
    device: Device,
    _marker: std::marker::PhantomData<E>,
}

impl<E: Environment> PPOAgent<E>
where
    E::State: ToTensor,
    E::Action: From<i64>,
{
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        pn_lr: f32,
        vn_lr: f32,
        gamma: f32,
        lambda: f32,
        clip_ratio: f32,
        device: Device,
    ) -> Self {
        let pn = MLP::new(device, state_dim, action_dim);
        let vn = MLP::new(device, state_dim, 1);

        let pn_opt = nn::Adam::default()
            .build(&pn.var_store, pn_lr.into())
            .unwrap();
        let vn_opt = nn::Adam::default()
            .build(&vn.var_store, vn_lr.into())
            .unwrap();

        Self {
            policy_net: pn,
            value_net: vn,
            pn_opt,
            vn_opt,
            pn_lr,
            vn_lr,
            gamma,
            lambda,
            clip_ratio,
            device,
            _marker: std::marker::PhantomData,
        }
    }
}
