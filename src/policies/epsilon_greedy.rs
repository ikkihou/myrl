use crate::environment::Environment;
use crate::policy::Policy;
use rand::Rng;

// pub struct EpsilonGreedyPolicy<E: Environment> {
//     epsilon: f32,
//     action_space: usize,
// }

// impl<E: Environment> Policy<E::State, E::Action> for EpsilonGreedyPolicy<E> {
//     fn select_action(&self, state: &E::State) -> E::Action {
//         let mut rng = rand::thread_rng();
//         if rng.r#gen::<f32>() < self.epsilon {
//             // 随机选择一个动作
//             rng.gen_range(0..self.action_space) as E::Action
//         } else {
//             // 选择最优动作
//             self.best_action(state)
//         }
//     }

//     fn explore(&mut self) {
//         // 逐渐减小 epsilon
//         self.epsilon *= 0.99;
//     }
// }
