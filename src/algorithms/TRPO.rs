use crate::agent::Agent;
use crate::environment::Environment;
use crate::networks::mlp::MLP;
use crate::policy::Policy;
use crate::utils::ToTensor;

use tch::nn::{Module, OptimizerConfig};
use tch::{nn, Device, Kind, Tensor};

const MAX_BATCH_STEPS: usize = 2048;
const VALUE_EPOCHS: usize = 10;
const VALUE_LR: f64 = 1e-3;
const CG_ITERS: usize = 10;
const LINE_SEARCH_STEPS: usize = 10;
const LINE_SEARCH_SHRINK: f64 = 0.5;

struct TrajectoryBatch {
    states: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    returns: Tensor,
    advantages: Tensor,
    episode_rewards: Vec<f32>,
}

pub struct TrpoAgent<E: Environment> {
    policy_net: MLP,
    value_net: MLP,
    value_opt: nn::Optimizer,
    gamma: f32,
    tau: f32,
    max_kl: f32,
    damping: f32,
    device: Device,
    _marker: std::marker::PhantomData<E>,
}

impl<E: Environment<Reward = f32>> TrpoAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    pub fn new(
        state_dim: usize,
        action_space: usize,
        gamma: f32,
        tau: f32,
        max_kl: f32,
        damping: f32,
        device: Device,
    ) -> Self {
        let policy_vs = nn::VarStore::new(device);
        let value_vs = nn::VarStore::new(device);

        let policy_net = MLP::new(policy_vs, state_dim, action_space);
        let value_net = MLP::new(value_vs, state_dim, 1);
        let value_opt = nn::Adam::default()
            .build(value_net.var_store(), VALUE_LR)
            .unwrap();

        Self {
            policy_net,
            value_net,
            value_opt,
            gamma,
            tau,
            max_kl,
            damping,
            device,
            _marker: std::marker::PhantomData,
        }
    }

    fn sample_action(&self, state: &E::State) -> (E::Action, Tensor) {
        let state_tensor = state.to_tensor().to_device(self.device);
        let logits = self.policy_net.model.forward(&state_tensor);
        let probs = logits.softmax(-1, Kind::Float);
        let action_tensor = probs.multinomial(1, true);
        let action = action_tensor.int64_value(&[0, 0]);
        let log_prob = logits
            .log_softmax(-1, Kind::Float)
            .gather(1, &action_tensor, false)
            .squeeze();
        (E::Action::from(action), log_prob.detach())
    }

    fn predict_value_tensor(&self, states: &Tensor) -> Tensor {
        self.value_net.model.forward(states).squeeze_dim(-1)
    }

    fn predict_value(&self, state: &E::State) -> f32 {
        let value =
            tch::no_grad(|| self.predict_value_tensor(&state.to_tensor().to_device(self.device)));
        value.double_value(&[0]) as f32
    }

    fn compute_advantages(&self, rewards: &[f32], values: &[f32], next_value: f32) -> Vec<f32> {
        let mut advantages = vec![0.0; rewards.len()];
        let mut gae = 0.0;

        for t in (0..rewards.len()).rev() {
            let next_v = if t + 1 < values.len() {
                values[t + 1]
            } else {
                next_value
            };

            let delta = rewards[t] + self.gamma * next_v - values[t];
            gae = delta + self.gamma * self.tau * gae;
            advantages[t] = gae;
        }

        advantages
    }

    fn normalize_advantages(advantages: Vec<f32>) -> Tensor {
        let advantages = Tensor::from_slice(&advantages);
        let mean = advantages.mean(Kind::Float);
        let std = advantages.std(true);
        ((advantages - mean) / (std + 1e-8)).to_kind(Kind::Float)
    }

    fn collect_trajectories(&self, env: &mut E) -> TrajectoryBatch {
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
                let step_result = env.step(&action);

                episode_reward += step_result.reward;
                episode_states.push(state_tensor);
                episode_actions.push(action.into());
                episode_log_probs.push(log_prob);
                episode_rewards_buffer.push(step_result.reward);
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
            let episode_advantages =
                self.compute_advantages(&episode_rewards_buffer, &episode_values, next_value);

            for (advantage, value) in episode_advantages.iter().zip(episode_values.iter()) {
                returns.push(advantage + value);
                all_advantages.push(*advantage);
            }

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

        TrajectoryBatch {
            states,
            actions,
            old_log_probs,
            returns,
            advantages,
            episode_rewards,
        }
    }

    fn flat_parameters(&self) -> Tensor {
        let params = self.policy_net.var_store.trainable_variables();
        Tensor::cat(
            &params
                .iter()
                .map(|param| param.detach().view([-1]))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn set_flat_parameters(&self, flat_params: &Tensor) {
        let params = self.policy_net.var_store.trainable_variables();
        let mut offset = 0;

        tch::no_grad(|| {
            for mut param in params {
                let numel = param.numel();
                let shape = param.size();
                let slice = flat_params
                    .narrow(0, offset as i64, numel as i64)
                    .view(shape.as_slice());
                param.copy_(&slice);
                offset += numel;
            }
        });
    }

    fn flat_grad(output: &Tensor, params: &[Tensor], create_graph: bool) -> Tensor {
        let grads = Tensor::run_backward(&[output], params, true, create_graph);
        Tensor::cat(
            &grads
                .into_iter()
                .map(|grad| {
                    if grad.defined() {
                        grad.contiguous().view([-1])
                    } else {
                        Tensor::zeros([0], (Kind::Float, output.device()))
                    }
                })
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn log_probs_from_logits(logits: &Tensor, actions: &Tensor) -> Tensor {
        logits
            .log_softmax(-1, Kind::Float)
            .gather(1, &actions.unsqueeze(-1), false)
            .squeeze_dim(-1)
    }

    fn surrogate_objective(
        &self,
        states: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        advantages: &Tensor,
    ) -> Tensor {
        let logits = self.policy_net.model.forward(states);
        let log_probs = Self::log_probs_from_logits(&logits, actions);
        let ratio = (log_probs - old_log_probs).exp();
        (ratio * advantages).mean(Kind::Float)
    }

    fn mean_kl(&self, states: &Tensor, old_logits: &Tensor) -> Tensor {
        let new_logits = self.policy_net.model.forward(states);
        let old_log_probs = old_logits.log_softmax(-1, Kind::Float);
        let new_log_probs = new_logits.log_softmax(-1, Kind::Float);
        let old_probs = old_log_probs.exp();
        (old_probs * (old_log_probs - new_log_probs))
            .sum_dim_intlist([-1].as_ref(), false, Kind::Float)
            .mean(Kind::Float)
    }

    fn fisher_vector_product(
        &self,
        states: &Tensor,
        old_logits: &Tensor,
        vector: &Tensor,
    ) -> Tensor {
        let params = self.policy_net.var_store.trainable_variables();
        let kl = self.mean_kl(states, old_logits);
        let flat_kl_grad = Self::flat_grad(&kl, &params, true);
        let kl_v = (&flat_kl_grad * vector).sum(Kind::Float);
        let flat_grad_grad = Self::flat_grad(&kl_v, &params, false);
        flat_grad_grad + vector * self.damping as f64
    }

    fn conjugate_gradient<F>(&self, fisher_product: F, b: &Tensor) -> Tensor
    where
        F: Fn(&Tensor) -> Tensor,
    {
        let mut x = Tensor::zeros_like(b);
        let mut r = b.shallow_clone();
        let mut p = r.shallow_clone();
        let mut rdotr = (&r * &r).sum(Kind::Float);

        for _ in 0..CG_ITERS {
            let z = fisher_product(&p);
            let alpha = &rdotr / ((&p * &z).sum(Kind::Float) + 1e-10);
            x += &alpha * &p;
            r -= &alpha * &z;

            let new_rdotr = (&r * &r).sum(Kind::Float);
            if new_rdotr.double_value(&[]) < 1e-10 {
                break;
            }

            let beta = &new_rdotr / (&rdotr + 1e-10);
            p = &r + &beta * &p;
            rdotr = new_rdotr;
        }

        x
    }

    fn line_search(
        &self,
        states: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        advantages: &Tensor,
        old_logits: &Tensor,
        old_params: &Tensor,
        full_step: &Tensor,
        old_objective: f64,
    ) -> bool {
        for step in 0..LINE_SEARCH_STEPS {
            let step_frac = LINE_SEARCH_SHRINK.powi(step as i32);
            let candidate = old_params + full_step * step_frac;
            self.set_flat_parameters(&candidate);

            let new_objective = self
                .surrogate_objective(states, actions, old_log_probs, advantages)
                .double_value(&[]);
            let new_kl = self.mean_kl(states, old_logits).double_value(&[]);

            if new_objective > old_objective && new_kl <= self.max_kl as f64 {
                return true;
            }
        }

        self.set_flat_parameters(old_params);
        false
    }

    fn update_value_function(&mut self, states: &Tensor, returns: &Tensor) {
        for _ in 0..VALUE_EPOCHS {
            let values = self.predict_value_tensor(states);
            let loss = (values - returns).pow_tensor_scalar(2.0).mean(Kind::Float);
            self.value_opt.backward_step(&loss);
        }
    }

    fn update_policy(&mut self, batch: &TrajectoryBatch) {
        let params = self.policy_net.var_store.trainable_variables();
        let old_params = self.flat_parameters();
        let old_logits = tch::no_grad(|| self.policy_net.model.forward(&batch.states));
        let objective = self.surrogate_objective(
            &batch.states,
            &batch.actions,
            &batch.old_log_probs,
            &batch.advantages,
        );
        let old_objective = objective.double_value(&[]);
        let gradient = Self::flat_grad(&objective, &params, false);

        if gradient.abs().max().double_value(&[]) < 1e-8 {
            return;
        }

        let fisher_product =
            |vector: &Tensor| self.fisher_vector_product(&batch.states, &old_logits, vector);
        let step_direction = self.conjugate_gradient(fisher_product, &gradient);
        let fisher_step = self.fisher_vector_product(&batch.states, &old_logits, &step_direction);
        let shs = 0.5
            * (&step_direction * fisher_step)
                .sum(Kind::Float)
                .double_value(&[]);

        if shs <= 0.0 {
            return;
        }

        let step_scale = (self.max_kl as f64 / (shs + 1e-10)).sqrt();
        let full_step = step_direction * step_scale;

        let _ = self.line_search(
            &batch.states,
            &batch.actions,
            &batch.old_log_probs,
            &batch.advantages,
            &old_logits,
            &old_params,
            &full_step,
            old_objective,
        );
    }

    fn update(&mut self, batch: &TrajectoryBatch) {
        self.update_value_function(&batch.states, &batch.returns);
        self.update_policy(batch);
    }
}

impl<E: Environment<Reward = f32>> Policy<E::State, E::Action> for TrpoAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    fn select_action(&self, state: &E::State) -> E::Action {
        self.sample_action(state).0
    }
}

impl<E: Environment<Reward = f32>> Agent<E> for TrpoAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    fn train(&mut self, env: &mut E, num_episodes: usize, if_plot: bool) {
        let mut all_rewards = Vec::with_capacity(num_episodes);

        for episode in 0..num_episodes {
            let batch = self.collect_trajectories(env);
            let mean_reward =
                batch.episode_rewards.iter().sum::<f32>() / batch.episode_rewards.len() as f32;
            println!("Episode {episode}: total reward = {mean_reward}");
            all_rewards.push(mean_reward);
            self.update(&batch);
        }

        if if_plot {
            crate::utils::plot_rewards(&all_rewards, "trpo_training.png", "Training Reward");
        }
    }

    fn train_episode(&mut self, _env: &mut E) -> f32 {
        unimplemented!("TRPO updates on trajectory batches, use train instead")
    }
}
