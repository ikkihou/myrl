use myrl::agent::Agent;
use myrl::algorithms::dqn::DqnAgent;
use myrl::environment::Environment;
use myrl::environments::cartpole::CartPole;

#[test]
fn dqn_cart_pole_test() {
    let device = tch::Device::Cpu;
    let mut env = CartPole::new();
    let mut agent = DqnAgent::<CartPole>::new(
        env.state_space(),
        env.action_space(),
        0.99,
        0.1,
        32,
        2,
        device,
    );
    agent.train(&mut env, 500, 10, true);
}
