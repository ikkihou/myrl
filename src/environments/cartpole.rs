use crate::environment::{Environment, StepResult};
use rand::random_range;

/// 简化版 CartPole 环境，状态维度为 [x, x_dot, theta, theta_dot]
pub struct CartPole {
    pub state: [f32; 4],
    pub step_limit: usize,
    pub step_count: usize,
}

impl CartPole {
    pub fn new() -> Self {
        Self {
            state: [0.0; 4],
            step_limit: 200,
            step_count: 0,
        }
    }

    fn is_done(&self, state: &[f32; 4]) -> bool {
        let x = state[0];
        let theta = state[2];
        x.abs() > 2.4 || theta.abs() > 12.0_f32.to_radians() || self.step_count >= self.step_limit
    }

    pub fn step(
        &mut self,
        action: &<CartPole as Environment>::Action,
    ) -> StepResult<<CartPole as Environment>::State, <CartPole as Environment>::Reward> {
        let mut state = self.state;

        // 动力学参数
        let g = 9.8; // 重力加速度
        let m_c = 1.0; // 小车质量
        let m_p = 0.1; // 杆质量
        let l = 0.5; // 杆长度
        let dt = 0.02; // 时间步长

        // 当前状态
        let x = state[0];
        let x_dot = state[1];
        let theta = state[2];
        let theta_dot = state[3];

        // 动作力
        let force = if *action == 1 { 10.0 } else { -10.0 };

        // 动力学方程
        let costheta = theta.cos();
        let sintheta = theta.sin();
        let temp = (force + m_p * l * theta_dot.powi(2) * sintheta) / (m_c + m_p);
        let theta_acc = (g * sintheta - costheta * temp)
            / (l * (4.0 / 3.0 - m_p * costheta.powi(2) / (m_c + m_p)));
        let x_acc = temp - m_p * l * theta_acc * costheta / (m_c + m_p);

        // 更新状态
        state[0] = x + dt * x_dot;
        state[1] = x_dot + dt * x_acc;
        state[2] = theta + dt * theta_dot;
        state[3] = theta_dot + dt * theta_acc;

        self.step_count += 1;
        self.state = state;

        // 判断是否结束
        let done = self.is_done(&state);

        // 奖励函数
        let reward = 1.0 - (x.abs() / 2.4 + theta.abs() / 12.0_f32.to_radians());

        StepResult {
            next_state: state,
            reward: if done { 0.0 } else { reward },
            done,
        }
    }
}

impl Environment for CartPole {
    type State = [f32; 4];
    type Action = i64; // 0: left, 1: right
    type Reward = f32;

    fn reset(&mut self) -> Self::State {
        // let mut rng = rng();
        self.state = [
            random_range(-0.05..0.05),
            random_range(-0.05..0.05),
            random_range(-0.05..0.05),
            random_range(-0.05..0.05),
        ];
        self.step_count = 0;
        self.state
    }

    fn step(&mut self, action: &Self::Action) -> StepResult<Self::State, Self::Reward> {
        self.step(action)
    }

    fn current_state(&self) -> Self::State {
        self.state
    }

    fn action_space(&self) -> usize {
        2 // 0: left, 1: right
    }

    fn state_dim(&self) -> usize {
        4 // [x, x_dot, theta, theta_dot]
    }
}
