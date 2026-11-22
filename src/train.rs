use crate::environment::Environment;

pub trait Train<E: Environment> {
    /// 训练 `num_episodes` 个 episode（TRPO 等批量算法为一次迭代），返回每次的总奖励
    fn train(&mut self, env: &mut E, num_episodes: usize) -> Vec<f32>;

    /// 训练一个episode，返回总奖励
    fn train_episode(&mut self, env: &mut E) -> f32;
}
