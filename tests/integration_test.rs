use myrl::agent::Agent;
use myrl::algorithms::a2c::A2CAgent;
use myrl::algorithms::dqn::DqnAgent;
use myrl::algorithms::policy_gradient::PGAgent;
use myrl::environment::Environment;
use myrl::environments::cartpole::CartPole;

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
    agent.train(&mut env, 1000, true);
}

#[test]
fn vpg_cart_pole_test() {
    let device = tch::Device::Cpu;
    let mut env = CartPole::new();
    let mut agent =
        PGAgent::<CartPole>::new(env.state_dim(), env.action_space(), 0.99, 0.1, 32, device);
    agent.train(&mut env, 1000, true);
}

#[test]
fn a2c_cart_pole_test() {
    let device = tch::Device::Cpu;
    let mut env = CartPole::new();
    let mut agent =
        A2CAgent::<CartPole>::new(env.state_dim(), env.action_space(), 0.99, 0.1, device);
    agent.train(&mut env, 1000, true);
}
