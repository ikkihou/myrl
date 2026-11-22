use std::collections::VecDeque;

use crate::environment::Environment;
use crate::networks::mlp::MLP;
use crate::policy::Policy;
use crate::train::Train;
use crate::utils::ToTensor;
use rand::{Rng, rng};
use tch::nn::{Module, OptimizerConfig};
use tch::{Device, Kind, Tensor, nn};

const MAX_REPLAY_BUFFER: usize = 10000;

type Transition<E> = (
    <E as Environment>::State,
    <E as Environment>::Action,
    f32,
    <E as Environment>::State,
    bool,
);

pub struct DqnAgent<E: Environment> {
    policy_net: MLP,
    target_net: MLP,
    replay_buffer: VecDeque<Transition<E>>,
    gamma: f32,
    epsilon: f32,
    batch_size: usize,
    action_space: usize,
    target_update_freq: usize,
    optimizer: nn::Optimizer,
    device: Device,
}

impl<E: Environment> DqnAgent<E>
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
        target_update_freq: usize,
        device: Device,
    ) -> Self {
        let policy_net = MLP::new(device, state_dim, action_space);
        let target_net = MLP::new(device, state_dim, action_space);
        let optimizer = nn::Adam::default()
            .build(&policy_net.var_store, 1e-3)
            .unwrap();

        DqnAgent {
            policy_net,
            target_net,
            replay_buffer: VecDeque::new(),
            gamma,
            epsilon,
            batch_size,
            action_space,
            target_update_freq,
            optimizer,
            device,
        }
    }

    fn update_policy_network(&mut self) {
        let buffer_len = self.replay_buffer.len();
        let indices: Vec<usize> = rand::seq::index::sample(&mut rng(), buffer_len, self.batch_size)
            .into_iter()
            .collect();
        let batch: Vec<_> = indices
            .into_iter()
            .map(|i| self.replay_buffer[i].clone())
            .collect();

        let mut rewards = Vec::with_capacity(self.batch_size);
        let mut dones = Vec::with_capacity(self.batch_size);
        let mut actions = Vec::with_capacity(self.batch_size);
        let mut state_tensors = Vec::with_capacity(self.batch_size);
        let mut next_state_tensors = Vec::with_capacity(self.batch_size);

        for (state, action, reward, next_state, done) in batch {
            rewards.push(reward);
            dones.push(if done { 1.0 } else { 0.0 });
            actions.push(action.into());
            state_tensors.push(state.to_tensor());
            next_state_tensors.push(next_state.to_tensor());
        }

        let state_batch = Tensor::stack(&state_tensors, 0)
            .squeeze_dim(1)
            .to(self.device);
        let next_state_batch = Tensor::stack(&next_state_tensors, 0)
            .squeeze_dim(1)
            .to(self.device);
        let action_batch = Tensor::from_slice(&actions)
            .to_device(self.device)
            .unsqueeze(1);
        let reward_batch = Tensor::from_slice(&rewards).to_device(self.device);
        let done_batch = Tensor::from_slice(&dones).to_device(self.device);

        // Q(s, a): 当前网络对实际执行动作的 Q 值
        let q_values = self
            .policy_net
            .model
            .forward(&state_batch)
            .gather(1, &action_batch, false)
            .squeeze_dim(1);

        // 目标值 r + gamma * max_a' Q_target(s', a')，无梯度
        let next_q_values = tch::no_grad(|| {
            self.target_net
                .model
                .forward(&next_state_batch)
                .max_dim(1, false)
                .0
        });
        let expected_q_values: Tensor =
            &reward_batch + self.gamma * next_q_values * (1.0 - done_batch);

        let loss = (q_values - expected_q_values.detach())
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float);
        self.optimizer.backward_step(&loss);
    }

    fn update_target_network(&mut self) {
        self.target_net
            .var_store
            .copy(&self.policy_net.var_store)
            .unwrap();
    }
}

impl<E: Environment> Policy<E::State, E::Action> for DqnAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    fn select_action(&self, state: &E::State) -> E::Action {
        let mut rng = rng();
        if rng.random::<f32>() < self.epsilon {
            E::Action::from(rng.random_range(0..self.action_space as i64))
        } else {
            let state_tensor = state.to_tensor().to(self.device);
            let action = tch::no_grad(|| {
                self.policy_net
                    .model
                    .forward(&state_tensor)
                    .argmax(1, false)
                    .int64_value(&[0])
            });
            E::Action::from(action)
        }
    }
}

impl<E: Environment> Train<E> for DqnAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    fn train(&mut self, env: &mut E, num_episodes: usize) -> Vec<f32> {
        let mut all_rewards = Vec::with_capacity(num_episodes);

        for episode in 0..num_episodes {
            let reward = self.train_episode(env);
            println!("Episode {episode}: total reward = {reward}");
            all_rewards.push(reward);

            if episode % self.target_update_freq == 0 {
                self.update_target_network();
            }
        }

        all_rewards
    }

    fn train_episode(&mut self, env: &mut E) -> f32 {
        let mut state = env.reset();
        let mut total_reward = 0.0;

        loop {
            let action = self.select_action(&state);
            let result = env.step(&action);
            total_reward += result.reward.into();

            if self.replay_buffer.len() >= MAX_REPLAY_BUFFER {
                self.replay_buffer.pop_front();
            }
            self.replay_buffer.push_back((
                state.clone(),
                action.clone(),
                result.reward.into(),
                result.next_state.clone(),
                result.done,
            ));

            if self.replay_buffer.len() >= self.batch_size {
                self.update_policy_network();
            }

            if result.done {
                break;
            }

            state = result.next_state.clone();
        }

        total_reward
    }
}
