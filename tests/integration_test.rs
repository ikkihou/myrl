use myrl::algorithms::a2c::A2CAgent;
use myrl::algorithms::dqn::DqnAgent;
use myrl::algorithms::policy_gradient::PGAgent;
use myrl::algorithms::ppo::PPOAgent;
use myrl::algorithms::trpo::TrpoAgent;
use myrl::environment::Environment;
use myrl::environments::cartpole::CartPole;
use myrl::plot::plot_rewards;
use myrl::train::Train;

#[test]
fn dqn_cart_pole_test() {
    let device = tch::Device::Cpu;
    let mut env = CartPole::new();
    let mut agent = DqnAgent::<CartPole>::new(
        env.state_dim(),
        env.action_space(),
        0.99,
        0.1,
        32,
        10,
        device,
    );
    let rewards = agent.train(&mut env, 1000);
    plot_rewards(&rewards, "dqn_training.png", "DQN CartPole Training");
}

#[test]
fn vpg_cart_pole_test() {
    let device = tch::Device::Cpu;
    let mut env = CartPole::new();
    let mut agent = PGAgent::<CartPole>::new(env.state_dim(), env.action_space(), 0.99, device);
    let rewards = agent.train(&mut env, 1000);
    plot_rewards(&rewards, "vpg_training.png", "VPG CartPole Training");
}

#[test]
fn a2c_cart_pole_test() {
    let device = tch::Device::Cpu;
    let mut env = CartPole::new();
    let mut agent = A2CAgent::<CartPole>::new(env.state_dim(), env.action_space(), 0.99, device);
    let rewards = agent.train(&mut env, 1000);
    plot_rewards(&rewards, "a2c_training.png", "A2C CartPole Training");
}

#[test]
fn trpo_cart_pole_test() {
    let device = tch::Device::Cpu;
    let mut env = CartPole::new();
    let mut agent = TrpoAgent::<CartPole>::new(
        env.state_dim(),
        env.action_space(),
        0.99,
        0.95,
        0.01,
        0.1,
        device,
    );
    let rewards = agent.train(&mut env, 200);
    plot_rewards(&rewards, "trpo_training.png", "TRPO CartPole Training");
}

#[test]
fn ppo_cart_pole_test() {
    let device = tch::Device::Cpu;
    let mut env = CartPole::new();
    let mut agent = PPOAgent::<CartPole>::new(
        env.state_dim(),
        env.action_space(),
        3e-4,
        1e-4,
        0.99,
        0.97,
        0.2,
        device,
    );
    let rewards = agent.train(&mut env, 200);
    plot_rewards(&rewards, "ppo_training.png", "PPO CartPole Training");
}
