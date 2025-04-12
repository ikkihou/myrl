use std::vec;

use crate::agent::Agent;
use crate::environment::Environment;
use crate::networks::mlp::MLP;
use crate::utils::ToTensor;

use serde::de::DeserializeOwned;
use tch::nn::{Module, OptimizerConfig};
use tch::{Device, IndexOp, Kind, Tensor, nn};

pub struct PGAgent<E: Environment> {
    policy_net: MLP,
    gamma: f32,
    epsilon: f32,
    batch_size: usize,
    action_dim: usize,
    optimizer: nn::Optimizer,
    device: Device,
    _marker: std::marker::PhantomData<E>, // ⬅️ 为了让编译器知道用到了 E
}

impl<E: Environment> PGAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    pub fn new(
        state_dim: usize,
        action_space: usize,
        gamma: f32,
        epsilon: f32,
        batch_size: usize,
        device: Device,
    ) -> Self {
        let vs = nn::VarStore::new(device);
        let policy_net = MLP::new(vs, state_dim, action_space);
        let opt = nn::Adam::default()
            .build(policy_net.var_store(), 1e-3)
            .unwrap();

        PGAgent {
            policy_net: policy_net,
            gamma: gamma,
            epsilon: epsilon,
            batch_size: batch_size,
            action_dim: action_space,
            optimizer: opt,
            device: device,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn select_action(&self, state: &E::State) -> E::Action {
        let state_tensor = state.clone().to_tensor(); // NOTE:测试是否需要clone
        let action_probs = self.policy_net.model.forward(&state_tensor).to(self.device);
        let action_probs = action_probs.softmax(-1, Kind::Float);
        let action = action_probs.multinomial(1, true).int64_value(&[0]);
        E::Action::from(action)
    }

    pub fn update_policy(&mut self, rewards: &Vec<E::Reward>, log_probs: &Vec<Tensor>) {
        let mut discounted_rewards: Vec<f32> = Vec::with_capacity(rewards.len());
        let mut r = 0.0;
        for v in rewards.iter().rev() {
            r = (*v).into() + self.gamma * r;
            discounted_rewards.insert(0, r);
        }
        let mean = discounted_rewards.iter().sum::<f32>() / discounted_rewards.len() as f32;
        let std = (discounted_rewards
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / discounted_rewards.len() as f32)
            .sqrt();
        let eps = 1e-9;
        let normalized_discount_rewards: Vec<f32> = discounted_rewards
            .iter()
            .map(|x| (x - mean) / (std + eps))
            .collect();
        let norm_discnt_rewards_tensors: Vec<Tensor> = normalized_discount_rewards
            .iter()
            .map(|&x| Tensor::from(x))
            .collect();

        let mut policy_loss: Vec<Tensor> = Vec::with_capacity(log_probs.len());
        for (log_prob, Gt) in log_probs.iter().zip(norm_discnt_rewards_tensors.iter()) {
            policy_loss.push(-log_prob * Gt);
        }
        let loss = Tensor::stack(&policy_loss, 0).sum(Kind::Float);

        self.optimizer.backward_step(&loss);
    }
}

impl<E: Environment> Agent<E> for PGAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    fn train(
        &mut self,
        env: &mut E,
        num_episodes: usize,
        target_update_freq: usize,
        if_plot: bool,
    ) {
        let mut all_rewards: Vec<f32> = Vec::with_capacity(num_episodes);

        for episode in 0..num_episodes {
            let reward = self.train_episode(env);
            println!("Episode {episode}: total reward = {reward}");
            all_rewards.push(reward);
        }

        if if_plot {
            crate::utils::plot_rewards(&all_rewards, "vpg_training.png", "Training Reward");
        }
    }

    fn train_episode(&mut self, env: &mut E) -> f32 {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut rewards: Vec<E::Reward> = Vec::new();
        let mut log_probs: Vec<Tensor> = Vec::new();
        loop {
            let action = self.select_action(&state);
            let step_result = env.step(&action);
            let next_state = step_result.next_state;
            let reward = step_result.reward;
            let done = step_result.done;

            total_reward += reward.clone().into();

            let probs = self
                .policy_net
                .model
                .forward(&state.to_tensor().to(self.device))
                .softmax(-1, Kind::Float);

            let action_idx = action.clone().into();
            let log_prob = probs.i((0, action_idx)).log();

            rewards.push(reward);
            log_probs.push(log_prob);

            if done {
                break;
            }

            state = next_state.clone();
        }
        self.update_policy(&rewards, &log_probs);
        total_reward
    }
}
