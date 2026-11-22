use crate::environment::Environment;
use crate::networks::mlp::MLP;
use crate::policy::Policy;
use crate::train::Train;
use crate::utils::ToTensor;

use tch::nn::{Module, OptimizerConfig};
use tch::{Device, Kind, Tensor, nn};

const MAX_BATCH_STEPS: usize = 2048;
const PPO_EPOCHS: usize = 10;
const MINIBATCH_SIZE: usize = 64;
const VALUE_EPOCHS: usize = 10;
const ENTROPY_COEF: f64 = 0.01;

struct GAEBuffer {
    states: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    returns: Tensor,
    advantages: Tensor,
    episode_rewards: Vec<f32>,
}

pub struct PPOAgent<E: Environment> {
    policy_net: MLP,
    value_net: MLP,
    pn_opt: nn::Optimizer,
    vn_opt: nn::Optimizer,
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
            gamma,
            lambda,
            clip_ratio,
            device,
            _marker: std::marker::PhantomData,
        }
    }

    fn sample_action(&self, state: &E::State) -> (i64, Tensor) {
        let state_tensor = state.to_tensor().to_device(self.device);
        let logits = self.policy_net.model.forward(&state_tensor);
        let action_tensor = logits.softmax(-1, Kind::Float).multinomial(1, true);
        let action = action_tensor.int64_value(&[0, 0]);
        let log_prob = logits
            .log_softmax(-1, Kind::Float)
            .gather(1, &action_tensor, false)
            .squeeze();
        (action, log_prob.detach())
    }

    fn predict_value(&self, state: &E::State) -> f32 {
        let state_tensor = state.to_tensor().to_device(self.device);
        let value = tch::no_grad(|| self.value_net.model.forward(&state_tensor));
        value.double_value(&[0]) as f32
    }

    /// GAE(lambda) 优势估计，返回 (advantages, returns)
    fn compute_gae(
        &self,
        rewards: &[f32],
        values: &[f32],
        next_value: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut advantages = vec![0.0; rewards.len()];
        let mut gae = 0.0;

        for t in (0..rewards.len()).rev() {
            let next_v = if t + 1 < values.len() {
                values[t + 1]
            } else {
                next_value
            };

            let delta = rewards[t] + self.gamma * next_v - values[t];
            gae = delta + self.gamma * self.lambda * gae;
            advantages[t] = gae;
        }

        let returns = advantages
            .iter()
            .zip(values.iter())
            .map(|(adv, value)| adv + value)
            .collect();

        (advantages, returns)
    }

    fn normalize_advantages(advantages: Vec<f32>) -> Tensor {
        let advantages = Tensor::from_slice(&advantages);
        let mean = advantages.mean(Kind::Float);
        let std = advantages.std(true);
        (advantages - mean) / (std + 1e-8)
    }

    fn collect_trajectories(&mut self, env: &mut E) -> GAEBuffer {
        let mut state_tensors = Vec::new();
        let mut actions = Vec::new();
        let mut log_probs = Vec::new();
        let mut returns = Vec::new();
        let mut all_advantages = Vec::new();
        let mut episode_rewards = Vec::new();
        let mut total_steps = 0;

        while total_steps < MAX_BATCH_STEPS {
            let mut state = env.reset();
            let mut episode_reward = 0.0;
            let mut episode_states = Vec::new();
            let mut episode_actions = Vec::new();
            let mut episode_log_probs = Vec::new();
            let mut episode_rewards_buffer = Vec::new();
            let mut episode_values = Vec::new();
            let mut done = false;

            while !done && total_steps < MAX_BATCH_STEPS {
                let state_tensor = state.to_tensor().to_device(self.device).squeeze_dim(0);
                let value = self.predict_value(&state);
                let (action, log_prob) = self.sample_action(&state);
                let env_action = E::Action::from(action);
                let step_result = env.step(&env_action);

                episode_reward += step_result.reward.into();
                episode_states.push(state_tensor);
                episode_actions.push(action);
                episode_log_probs.push(log_prob);
                episode_rewards_buffer.push(step_result.reward.into());
                episode_values.push(value);

                state = step_result.next_state;
                done = step_result.done;
                total_steps += 1;
            }

            let next_value = if done {
                0.0
            } else {
                self.predict_value(&state)
            };
            let (episode_advantages, episode_returns) =
                self.compute_gae(&episode_rewards_buffer, &episode_values, next_value);

            all_advantages.extend(episode_advantages);
            returns.extend(episode_returns);

            state_tensors.extend(episode_states);
            actions.extend(episode_actions);
            log_probs.extend(episode_log_probs);
            episode_rewards.push(episode_reward);
        }

        let states = Tensor::stack(&state_tensors, 0).to_device(self.device);
        let actions = Tensor::from_slice(&actions)
            .to_kind(Kind::Int64)
            .to_device(self.device);
        let old_log_probs = Tensor::stack(&log_probs, 0).to_device(self.device);
        let returns = Tensor::from_slice(&returns).to_device(self.device);
        let advantages = Self::normalize_advantages(all_advantages).to_device(self.device);

        GAEBuffer {
            states,
            actions,
            old_log_probs,
            returns,
            advantages,
            episode_rewards,
        }
    }

    fn log_probs_from_logits(logits: &Tensor, actions: &Tensor) -> Tensor {
        logits
            .log_softmax(-1, Kind::Float)
            .gather(1, &actions.unsqueeze(-1), false)
            .squeeze_dim(-1)
    }

    /// Clipped surrogate objective: -min(ratio * A, clip(ratio) * A)
    fn clipped_surrogate_loss(
        &self,
        states: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        advantages: &Tensor,
    ) -> Tensor {
        let logits = self.policy_net.model.forward(states);
        let log_probs = Self::log_probs_from_logits(&logits, actions);
        let ratio = (log_probs - old_log_probs).exp();
        let clipped_ratio = ratio.clamp(1.0 - self.clip_ratio as f64, 1.0 + self.clip_ratio as f64);
        let surr = ratio * advantages;
        let clipped_surr = clipped_ratio * advantages;
        surr.minimum(&clipped_surr).mean(Kind::Float).neg()
    }

    fn update_policy(&mut self, batch: &GAEBuffer) {
        let batch_size = batch.states.size()[0] as usize;
        let mut indices: Vec<i64> = (0..batch_size as i64).collect();

        for _ in 0..PPO_EPOCHS {
            // 随机打乱索引，切分 minibatch
            shuffle(&mut indices);
            for minibatch in indices.chunks(MINIBATCH_SIZE) {
                let idx = Tensor::from_slice(minibatch).to_device(self.device);
                let mb_states = batch.states.index_select(0, &idx);
                let mb_actions = batch.actions.index_select(0, &idx);
                let mb_old_log_probs = batch.old_log_probs.index_select(0, &idx);
                let mb_advantages = batch.advantages.index_select(0, &idx);

                let loss = self.clipped_surrogate_loss(
                    &mb_states,
                    &mb_actions,
                    &mb_old_log_probs,
                    &mb_advantages,
                );
                let logits = self.policy_net.model.forward(&mb_states);
                let log_probs_all = logits.log_softmax(-1, Kind::Float);
                let probs = log_probs_all.exp();
                let entropy = -(probs * log_probs_all)
                    .sum_dim_intlist([-1].as_ref(), false, Kind::Float)
                    .mean(Kind::Float);
                let loss = loss - ENTROPY_COEF * entropy;

                self.pn_opt.backward_step(&loss);
            }
        }
    }

    fn update_value_function(&mut self, batch: &GAEBuffer) {
        let batch_size = batch.states.size()[0] as usize;
        let mut indices: Vec<i64> = (0..batch_size as i64).collect();

        for _ in 0..VALUE_EPOCHS {
            shuffle(&mut indices);
            for minibatch in indices.chunks(MINIBATCH_SIZE) {
                let idx = Tensor::from_slice(minibatch).to_device(self.device);
                let mb_states = batch.states.index_select(0, &idx);
                let mb_returns = batch.returns.index_select(0, &idx).squeeze_dim(-1);

                let values = self.value_net.model.forward(&mb_states).squeeze_dim(-1);
                let loss = (values - mb_returns)
                    .pow_tensor_scalar(2.0)
                    .mean(Kind::Float);

                self.vn_opt.backward_step(&loss);
            }
        }
    }
}

/// Fisher-Yates shuffle（避免引入 rand 依赖）
fn shuffle(indices: &mut [i64]) {
    let (mut seed, _) = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_nanos() as u64, ()))
        .unwrap_or((0x9E3779B97F4A7C15, ()));

    let mut next_u64 = move || {
        // xorshift64
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for i in (1..indices.len()).rev() {
        let j = (next_u64() % (i as u64 + 1)) as usize;
        indices.swap(i, j);
    }
}

impl<E: Environment> Policy<E::State, E::Action> for PPOAgent<E>
where
    E::State: ToTensor,
    E::Action: From<i64>,
{
    fn select_action(&self, state: &E::State) -> E::Action {
        E::Action::from(self.sample_action(state).0)
    }
}

impl<E: Environment> Train<E> for PPOAgent<E>
where
    E::State: ToTensor,
    E::Action: From<i64>,
{
    fn train(&mut self, env: &mut E, num_episodes: usize) -> Vec<f32> {
        let mut all_rewards = Vec::with_capacity(num_episodes);

        for episode in 0..num_episodes {
            let batch = self.collect_trajectories(env);
            let mean_reward =
                batch.episode_rewards.iter().sum::<f32>() / batch.episode_rewards.len() as f32;
            println!("Episode {episode}: total reward = {mean_reward}");
            all_rewards.push(mean_reward);
            self.update_policy(&batch);
            self.update_value_function(&batch);
        }

        all_rewards
    }

    fn train_episode(&mut self, env: &mut E) -> f32 {
        let batch = self.collect_trajectories(env);
        let mean_reward =
            batch.episode_rewards.iter().sum::<f32>() / batch.episode_rewards.len() as f32;
        self.update_policy(&batch);
        self.update_value_function(&batch);
        mean_reward
    }
}
