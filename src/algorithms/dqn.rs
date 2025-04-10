use std::collections::VecDeque;

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
    action_dim: usize,
    optimizer: nn::Optimizer,
    device: Device,
}

impl<E: Environment> DqnAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        gamma: f32,
        epsilon: f32,
        batch_size: usize,
        action_dim: usize,
        device: Device,
    ) -> Self {
        let vs1 = nn::VarStore::new(device);
        let vs2 = nn::VarStore::new(device);

        let policy_net = MLP::new(vs1, input_dim, output_dim);
        let target_net = MLP::new(vs2, input_dim, output_dim);

        let opt = nn::Adam::default()
            .build(policy_net.var_store(), 1e-4)
            .unwrap();

        DqnAgent {
            policy_net: policy_net,
            target_net: target_net,
            replay_buffer: VecDeque::new(),
            gamma,
            epsilon,
            batch_size,
            action_dim,
            optimizer: opt,
            device,
        }
    }

    fn update_policy_network(&mut self) {
        use tch::Kind;

        // sampling minibatch
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

        // 转 Tensor
        let state_batch = Tensor::stack(
            &states
                .into_iter()
                .map(|s| s.to_tensor())
                .collect::<Vec<_>>(),
            0,
        )
        .squeeze_dim(1)
        .to(self.device);
        // println!("state_batch shape: {:?}", state_batch.dim());
        // println!("{:?}", state_batch);

        let next_state_batch = Tensor::stack(
            &next_states
                .into_iter()
                .map(|s| s.to_tensor())
                .collect::<Vec<_>>(),
            0,
        )
        .squeeze_dim(1)
        .to(self.device);
        // println!("next_state_batch shape: {:?}", next_state_batch.dim());
        // println!("{:?}", next_state_batch);

        let action_batch =
            Tensor::from_slice(&actions.into_iter().map(|a| a.into()).collect::<Vec<_>>())
                .to_device(self.device)
                .unsqueeze(1);
        // println!("action_batch shape: {:?}", action_batch.dim());
        // println!("{:?}", action_batch);

        let reward_batch = Tensor::from_slice(&rewards)
            .to_device(self.device)
            .unsqueeze(1);
        // println!("reward_batch shape: {:?}", reward_batch.dim());
        // println!("{:?}", reward_batch);

        let done_batch = Tensor::from_slice(
            &dones
                .iter()
                .map(|&d| if d { 0.0 } else { 1.0 })
                .collect::<Vec<_>>(),
        )
        .to_device(self.device)
        .unsqueeze(1);
        // println!("done_batch shape: {:?}", done_batch.dim());
        // println!("{:?}", done_batch);

        // current q values
        let q_values = self
            .policy_net
            .model
            .forward(&state_batch)
            .gather(1, &action_batch, false);

        // target q values
        let next_q_values = self
            .target_net
            .model
            .forward(&next_state_batch)
            .max_dim(1, false)
            .0; // Extract the tensor from the tuple
        let expected_q_values: Tensor =
            &reward_batch + self.gamma * next_q_values * (1 - done_batch);

        let loss = (q_values - expected_q_values.detach())
            .pow(&Tensor::from(2))
            .mean(Kind::Float);

        self.optimizer.backward_step(&loss);
    }

    fn update_target_network(&mut self) {
        use tch::no_grad;

        let policy_vars = self.policy_net.var_store().variables();
        let target_vars = self.target_net.var_store().variables();

        no_grad(|| {
            for (name, mut target_var) in target_vars {
                if let Some(policy_var) = policy_vars.get(&name) {
                    // detach + move to same device just in case
                    let detached = policy_var.detach().to_device(target_var.device());
                    target_var.copy_(&detached);
                }
            }
        });
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
            // 探索
            let action = Rng::random_range(&mut rng, 0..self.action_dim as i64);
            E::Action::from(action)
        } else {
            let state = &state;
            let _no_grad_gurad = tch::no_grad_guard();
            let action_tensor = self
                .policy_net
                .model
                .forward(&state.to_tensor().to(self.device));
            let action: i64 = action_tensor.argmax(1, false).int64_value(&[0]);
            E::Action::from(action)
        }
    }

    // fn explore(&self, state: &E::State) -> E::Action {
    //     todo!()
    // }
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

            // 更新 target 网络（通常每 fixed_target_update_freq 个 episode 更新一次）
            if episode % target_update_freq == 0 {
                self.update_target_network();
            }
        }

        // plot training rewards along episodes
        if if_plot {
            use plotters::prelude::*;

            let root = BitMapBackend::new("dqn_training.png", (640, 480)).into_drawing_area();
            root.fill(&WHITE).unwrap();
            let mut chart = ChartBuilder::on(&root)
                .caption("Training Reward", ("sans-serif", 30))
                .margin(20)
                .x_label_area_size(30)
                .y_label_area_size(40)
                .build_cartesian_2d(
                    0..all_rewards.len(),
                    0f32..all_rewards.iter().cloned().fold(0., f32::max),
                )
                .unwrap();

            chart.configure_mesh().draw().unwrap();

            chart
                .draw_series(LineSeries::new(
                    all_rewards.iter().enumerate().map(|(i, r)| (i, *r)),
                    &RED,
                ))
                .unwrap()
                .label("Reward")
                .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED));

            chart.configure_series_labels().draw().unwrap();
            println!("Saved training plot to dqn_training.png");
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
                action,
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

    // fn evaluate(&self, env: &mut E) -> f32 {
    //     todo!()
    // }
}
