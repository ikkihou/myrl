use crate::environment::Environment;
use crate::networks::mlp::MLP;
use crate::policy::Policy;
use crate::train::Train;
use crate::utils::ToTensor;

use tch::nn::{Module, OptimizerConfig};
use tch::{Device, IndexOp, Kind, Tensor, nn};

pub struct A2CAgent<E: Environment> {
    actor_net: MLP,
    critic_net: MLP,
    actor_opt: nn::Optimizer,
    critic_opt: nn::Optimizer,
    gamma: f32,
    device: Device,
    _marker: std::marker::PhantomData<E>,
}

impl<E: Environment> A2CAgent<E>
where
    E::State: ToTensor,
    E::Action: From<i64>,
{
    pub fn new(state_dim: usize, action_space: usize, gamma: f32, device: Device) -> Self {
        let actor_net = MLP::new(device, state_dim, action_space);
        let critic_net = MLP::new(device, state_dim, 1);

        let actor_opt = nn::Adam::default()
            .build(&actor_net.var_store, 1e-4)
            .unwrap();
        let critic_opt = nn::Adam::default()
            .build(&critic_net.var_store, 1e-4)
            .unwrap();

        A2CAgent {
            actor_net,
            critic_net,
            actor_opt,
            critic_opt,
            gamma,
            device,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<E: Environment> Policy<E::State, E::Action> for A2CAgent<E>
where
    E::State: ToTensor,
    E::Action: From<i64>,
{
    fn select_action(&self, state: &E::State) -> E::Action {
        let state_tensor = state.to_tensor().to(self.device);
        let logits = self.actor_net.model.forward(&state_tensor);
        let action = logits
            .softmax(-1, Kind::Float)
            .multinomial(1, true)
            .int64_value(&[0]);
        E::Action::from(action)
    }
}

impl<E: Environment> Train<E> for A2CAgent<E>
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

        loop {
            let state_tensor = state.to_tensor().to(self.device);
            let logits = self.actor_net.model.forward(&state_tensor);
            let action = logits
                .softmax(-1, Kind::Float)
                .multinomial(1, true)
                .int64_value(&[0]);

            let result = env.step(&E::Action::from(action));
            total_reward += result.reward.into();

            // TD target: r + gamma * V(s')，终止时为 r
            let next_state_value = if result.done {
                Tensor::from(0.0)
            } else {
                let next_state_tensor = result.next_state.to_tensor().to(self.device);
                tch::no_grad(|| self.critic_net.model.forward(&next_state_tensor))
            };
            let td_target = tch::no_grad(|| {
                &Tensor::from(result.reward.into() as f64) + self.gamma * next_state_value
            });

            let state_value = self.critic_net.model.forward(&state_tensor);
            let advantage = &td_target.detach() - &state_value;

            let log_prob = logits.log_softmax(-1, Kind::Float).i((0, action));
            let actor_loss = (-log_prob * advantage.detach()).mean(Kind::Float);
            let critic_loss = (&state_value - td_target.detach())
                .pow_tensor_scalar(2.0)
                .mean(Kind::Float);

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
