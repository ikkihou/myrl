use crate::agent::Agent;
use crate::environment::Environment;
use crate::networks::mlp::MLP;
use crate::utils::ToTensor;

use tch::nn::{Module, OptimizerConfig};
use tch::{nn, Device, Kind, Tensor};

pub struct A2CAgent<E: Environment> {
    // NOTE: more fields to be added
    actor_net: MLP,
    critic_net: MLP,
    actor_opt: nn::Optimizer,
    critic_opt: nn::Optimizer,
    gamma: f32,
    epsilon: f32,
    device: Device,
    _marker: std::marker::PhantomData<E>, // ⬅️ 为了让编译器知道用到了 E
}

impl<E: Environment> A2CAgent<E>
where
    E::State: Clone + ToTensor,
    E::Action: Into<i64> + From<i64> + Clone,
{
    pub fn new(
        state_dim: usize,
        action_space: usize,
        gamma: f32,
        epsilon: f32,
        device: Device,
    ) -> Self {
        let actor_vs = nn::VarStore::new(device);
        let critic_vs = nn::VarStore::new(device);

        let actor_net = MLP::new(actor_vs, state_dim, action_space);
        let critic_net = MLP::new(critic_vs, state_dim, 1);

        let actor_opt = nn::Adam::default()
            .build(actor_net.var_store(), 1e-4)
            .unwrap();
        let critic_opt = nn::Adam::default()
            .build(critic_net.var_store(), 1e-4)
            .unwrap();

        A2CAgent {
            actor_net,
            critic_net,
            actor_opt,
            critic_opt,
            gamma,
            epsilon,
            device,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<E: Environment> Agent<E> for A2CAgent<E>
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
            crate::utils::plot_rewards(&all_rewards, "a2c_training.png", "Training Reward");
        }
    }

    fn train_episode(&mut self, env: &mut E) -> f32 {
        let mut state = env.reset();
        let mut total_reward = 0.0;

        loop {
            let state_tensor = state.to_tensor();
            // choose action
            let action_probs = self
                .actor_net
                .model
                .forward(&state_tensor.to(self.device))
                .softmax(-1, Kind::Float);
            let action = action_probs.multinomial(1, true).int64_value(&[0]);

            // execute action
            let result = env.step(&E::Action::from(action));
            let next_state = result.next_state.to_tensor();
            total_reward += result.reward.into();

            // calculate TD target
            let state_value = self.critic_net.model.forward(&state_tensor);
            let next_state_value = if result.done {
                Tensor::from(0.0)
            } else {
                tch::no_grad(|| self.critic_net.model.forward(&next_state))
            };

            let td_target = Tensor::from(result.reward.into()) + self.gamma * next_state_value;
            let advantage = td_target.shallow_clone() - state_value.shallow_clone();

            // calculate actor loss
            let log_prob = action_probs.log();
            let actor_loss = (-log_prob * advantage.detach()).mean(Kind::Float);

            // calculate critic loss
            let critic_loss = (state_value - td_target.detach())
                .pow_tensor_scalar(2.0)
                .mean(Kind::Float);

            // back propagation
            self.actor_opt.backward_step(&actor_loss);
            self.critic_opt.backward_step(&critic_loss);

            if result.done {
                break;
            }

            state = result.next_state;
        }

        total_reward
    }
}
