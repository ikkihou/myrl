use crate::agent::Agent;
use crate::environment::Environment;
use crate::networks::mlp::MLP;
use crate::utils::ToTensor;

use tch::nn::{Module, OptimizerConfig};
use tch::{nn, Device, IndexOp, Kind, Tensor};

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
            policy_net,
            gamma,
            epsilon,
            batch_size,
            action_dim: action_space,
            optimizer: opt,
            device,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn select_action(&self, state: &E::State) -> (E::Action, Tensor) {
        let state_tensor = state.to_tensor().to(self.device); // 添加 batch 维度
        let action_probs = self
            .policy_net
            .model
            .forward(&state_tensor)
            .softmax(-1, Kind::Float);
        let action = action_probs.multinomial(1, true).int64_value(&[0]);
        let log_prob = action_probs.i((0, action)).log();
        (E::Action::from(action), log_prob)
    }

    pub fn update_policy(&mut self, rewards: &[E::Reward], log_probs: &[Tensor]) {
        // 计算折扣奖励并归一化
        let mut discounted_rewards = Vec::with_capacity(rewards.len());
        let mut r = 0.0;
        for &reward in rewards.iter().rev() {
            r = reward.into() + self.gamma * r;
            discounted_rewards.push(r);
        }
        discounted_rewards.reverse();

        let mean = discounted_rewards.iter().sum::<f32>() / discounted_rewards.len() as f32;
        let std = (discounted_rewards
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / discounted_rewards.len() as f32)
            .sqrt();
        let eps = 1e-9;
        let normalized_rewards: Vec<_> = discounted_rewards
            .iter()
            .map(|x| (x - mean) / (std + eps))
            .collect();

        // 计算策略损失
        let policy_loss: Tensor = log_probs
            .iter()
            .zip(normalized_rewards.iter())
            .map(|(log_prob, &reward)| -log_prob * Tensor::from(reward)) // 修复点
            .collect::<Vec<_>>()
            .into_iter()
            .reduce(|acc, x| acc + x)
            .unwrap();

        self.optimizer.backward_step(&policy_loss);
    }
}

impl<E: Environment> Agent<E> for PGAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    fn train(&mut self, env: &mut E, num_episodes: usize, if_plot: bool) {
        let mut all_rewards: Vec<f32> = Vec::with_capacity(num_episodes);

        for episode in 0..num_episodes {
            let reward = self.train_episode(env);
            println!("Episode {episode}: total reward = {reward}");
            all_rewards.push(reward);

            // 动态调整 epsilon
            self.epsilon = (self.epsilon * 0.995).max(0.01);
        }

        if if_plot {
            crate::utils::plot_rewards(&all_rewards, "vpg_training.png", "Training Reward");
        }
    }

    fn train_episode(&mut self, env: &mut E) -> f32 {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut rewards = Vec::new();
        let mut log_probs = Vec::new();

        loop {
            let (action, log_prob) = self.select_action(&state);
            let step_result = env.step(&action);

            total_reward += step_result.reward.into();
            rewards.push(step_result.reward);
            log_probs.push(log_prob);

            if step_result.done {
                break;
            }

            state = step_result.next_state;
        }

        self.update_policy(&rewards, &log_probs);
        total_reward
    }
}
