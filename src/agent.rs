use crate::environment::Environment;

pub trait Agent<E: Environment> {
    fn train(&mut self, env: &mut E, num_episodes: usize, if_plot: bool);

    // 训练一个episode，返回总奖励
    fn train_episode(&mut self, env: &mut E) -> f32;

    // 评估当前策略，返回总奖励
    // fn evaluate(&self, env: &mut E) -> f32;

    // fn save(&self, path: &str) -> anyhow::Result<()>;

    // fn load(&mut self, path: &str) -> anyhow::Result<()>;
}
