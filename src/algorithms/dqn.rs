use std::collections::VecDeque;
use std::os::macos::raw::stat;

use crate::agent::Agent;
use crate::environment::Environment;
use crate::networks::mlp::MLP;
use crate::policy::Policy;
use crate::utils::ToTensor;

use rand::seq::IndexedRandom;
use rand::{Rng, rng};
use tch::nn::OptimizerConfig;
use tch::{Device, nn::Module};
use tch::{Tensor, nn};

const MAX_REPLAY_BUFFER: usize = 10000;
pub struct DqnAgent<E: Environment> {
    policy_net: MLP,
    target_net: MLP,
    replay_buffer: VecDeque<(E::State, E::Action, f32, E::State, bool)>,
    gamma: f32,
    epsilon: f32,
    batch_size: usize,
    action_space: usize,
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
        device: Device,
    ) -> Self {
        let vs1 = nn::VarStore::new(device);
        let vs2 = nn::VarStore::new(device);

        let policy_net = MLP::new(vs1, state_dim, action_space);
        let target_net = MLP::new(vs2, state_dim, action_space);

        let opt = nn::Adam::default()
            .build(policy_net.var_store(), 1e-3)
            .unwrap();

        DqnAgent {
            policy_net,
            target_net,
            replay_buffer: VecDeque::new(),
            gamma,
            epsilon,
            batch_size,
            action_space,
            optimizer: opt,
            device,
        }
    }

    fn update_policy_network(&mut self) {
        use tch::Kind;

        // Sample minibatch
        let minibatch: Vec<_> = self
            .replay_buffer
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng(), self.batch_size)
            .cloned()
            .collect();

        let mut states = Vec::with_capacity(self.batch_size);
        let mut actions = Vec::with_capacity(self.batch_size);
        let mut rewards = Vec::with_capacity(self.batch_size);
        let mut next_states = Vec::with_capacity(self.batch_size);
        let mut dones = Vec::with_capacity(self.batch_size);

        for (s, a, r, s_, d) in minibatch {
            states.push(s);
            actions.push(a);
            rewards.push(r);
            next_states.push(s_);
            dones.push(d);
        }

        // Convert to tensors
        let state_batch = Tensor::stack(
            &states
                .into_iter()
                .map(|s| s.to_tensor())
                .collect::<Vec<_>>(),
            0,
        )
        .squeeze_dim(1)
        .to(self.device);

        let next_state_batch = Tensor::stack(
            &next_states
                .into_iter()
                .map(|s| s.to_tensor())
                .collect::<Vec<_>>(),
            0,
        )
        .squeeze_dim(1)
        .to(self.device);

        let action_batch =
            Tensor::from_slice(&actions.into_iter().map(|a| a.into()).collect::<Vec<_>>())
                .to_device(self.device)
                .unsqueeze(1);

        let reward_batch = Tensor::from_slice(&rewards)
            .to_device(self.device)
            .unsqueeze(1);

        let done_batch = Tensor::from_slice(
            &dones
                .iter()
                .map(|&d| if d { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
        )
        .to_device(self.device)
        .unsqueeze(1);

        // Compute current Q values
        let q_values = self
            .policy_net
            .model
            .forward(&state_batch)
            .gather(1, &action_batch, false);

        // Compute target Q values (no gradient)
        let next_q_values = self
            .target_net
            .model
            .forward(&next_state_batch)
            .max_dim(1, false)
            .0;

        let expected_q_values: Tensor =
            &reward_batch + self.gamma * next_q_values.unsqueeze(1) * (1.0 - done_batch);

        // Compute loss
        let loss = (q_values - expected_q_values.detach())
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float);

        self.optimizer.backward_step(&loss);
    }

    fn update_target_network(&mut self) {
        let policy_state = self.policy_net.var_store();
        self.target_net.var_store.copy(policy_state).unwrap();
    }
}

impl<E: Environment> Policy<E::State, E::Action> for DqnAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    fn select_action(&self, state: &E::State) -> E::Action {
        let mut rng = rng();
        if rand::random::<f32>() < self.epsilon {
            let action = Rng::random_range(&mut rng, 0..self.action_space as i64);

            E::Action::from(action)
        } else {
            let state_tensor = &state.to_tensor();
            let _no_grad_gurad = tch::no_grad_guard();
            let action_tensor = self.policy_net.model.forward(&state_tensor.to(self.device));
            let action: i64 = action_tensor.argmax(1, false).int64_value(&[0]);

            E::Action::from(action)
        }
    }
}

impl<E: Environment> Agent<E> for DqnAgent<E>
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

            if episode % target_update_freq == 0 {
                self.update_target_network();
            }
        }

        // plot training rewards along episodes
        if if_plot {
            crate::utils::plot_rewards(&all_rewards, "dqn_training.png", "Training Reward");
        }
    }

    fn train_episode(&mut self, env: &mut E) -> f32 {
        let mut state = env.reset();
        let mut total_reward = 0.0;

        loop {
            // select action
            let action = self.select_action(&state);

            // execute action
            let stepresult = env.step(&action);
            total_reward += stepresult.reward.into();

            // store memory
            if self.replay_buffer.len() >= MAX_REPLAY_BUFFER {
                self.replay_buffer.pop_front();
            }
            self.replay_buffer.push_back((
                state.clone(),
                action.clone(),
                stepresult.reward.into(),
                stepresult.next_state.clone(),
                stepresult.done,
            ));

            // update network
            if self.replay_buffer.len() >= self.batch_size {
                self.update_policy_network();
            }

            if stepresult.done {
                break;
            }

            state = stepresult.next_state.clone();
        }

        total_reward
    }
}
