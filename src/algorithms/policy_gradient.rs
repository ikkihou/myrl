use crate::environment::Environment;
use crate::networks::mlp::MLP;
use crate::policy::Policy;
use crate::train::Train;
use crate::utils::ToTensor;

use tch::nn::{Module, OptimizerConfig};
use tch::{Device, IndexOp, Kind, Tensor, nn};

pub struct PGAgent<E: Environment> {
    policy_net: MLP,
    gamma: f32,
    optimizer: nn::Optimizer,
    device: Device,
    _marker: std::marker::PhantomData<E>,
}

impl<E: Environment> PGAgent<E>
where
    E::State: ToTensor,
    E::Action: From<i64>,
{
    pub fn new(state_dim: usize, action_space: usize, gamma: f32, device: Device) -> Self {
        let policy_net = MLP::new(device, state_dim, action_space);
        let optimizer = nn::Adam::default()
            .build(&policy_net.var_store, 1e-3)
            .unwrap();

        PGAgent {
            policy_net,
            gamma,
            optimizer,
            device,
            _marker: std::marker::PhantomData,
        }
    }

    fn sample(&self, state: &E::State) -> (E::Action, Tensor) {
        let state_tensor = state.to_tensor().to(self.device);
        let logits = self.policy_net.model.forward(&state_tensor);
        let action = logits
            .softmax(-1, Kind::Float)
            .multinomial(1, true)
            .int64_value(&[0]);
        let log_prob = logits.log_softmax(-1, Kind::Float).i((0, action));
        (E::Action::from(action), log_prob)
    }

    fn update_policy(&mut self, rewards: &[f32], log_probs: &[Tensor]) {
        // 折扣回报
        let mut discounted_rewards = Vec::with_capacity(rewards.len());
        let mut r = 0.0;
        for &reward in rewards.iter().rev() {
            r = reward + self.gamma * r;
            discounted_rewards.push(r);
        }
        discounted_rewards.reverse();

        // 基线归一化
        let mean = discounted_rewards.iter().sum::<f32>() / discounted_rewards.len() as f32;
        let var = discounted_rewards
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / discounted_rewards.len() as f32;
        let std = var.sqrt();

        let returns = (Tensor::from_slice(&discounted_rewards).to(self.device) - (mean as f64))
            / (std as f64 + 1e-9);

        let log_probs = Tensor::stack(log_probs, 0).to(self.device);
        let policy_loss = (-log_probs * returns).mean(Kind::Float);

        self.optimizer.backward_step(&policy_loss);
    }
}

impl<E: Environment> Policy<E::State, E::Action> for PGAgent<E>
where
    E::State: ToTensor,
    E::Action: From<i64>,
{
    fn select_action(&self, state: &E::State) -> E::Action {
        self.sample(state).0
    }
}

impl<E: Environment> Train<E> for PGAgent<E>
where
    E::State: ToTensor,
    E::Action: From<i64>,
{
    fn train(&mut self, env: &mut E, num_episodes: usize) -> Vec<f32> {
        let mut all_rewards = Vec::with_capacity(num_episodes);

        for episode in 0..num_episodes {
            let reward = self.train_episode(env);
            println!("Episode {episode}: total reward = {reward}");
            all_rewards.push(reward);
        }

        all_rewards
    }

    fn train_episode(&mut self, env: &mut E) -> f32 {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut rewards = Vec::new();
        let mut log_probs = Vec::new();

        loop {
            let (action, log_prob) = self.sample(&state);
            let result = env.step(&action);

            total_reward += result.reward.into();
            rewards.push(result.reward.into());
            log_probs.push(log_prob);

            if result.done {
                break;
            }

            state = result.next_state;
        }

        self.update_policy(&rewards, &log_probs);
        total_reward
    }
}
