pub struct StepResult<S, R> {
    pub next_state: S,
    pub reward: R,
    pub done: bool,
}

pub trait Environment {
    type State;
    type Action;
    type Reward: Copy + Into<f32>;

    fn reset(&mut self) -> Self::State;
    fn step(&mut self, action: &Self::Action) -> StepResult<Self::State, Self::Reward>; // (next_state, reward, if_done)

    /// 获取当前状态
    fn current_state(&self) -> Self::State;

    /// 动作空间维度
    fn action_space(&self) -> usize;

    /// 状态空间维度
    fn state_dim(&self) -> usize;
}
