use myrl::algorithms::a2c::A2CAgent;
use myrl::algorithms::dqn::DqnAgent;
use myrl::algorithms::policy_gradient::PGAgent;
use myrl::algorithms::trpo::TrpoAgent;
use myrl::environment::Environment;
use myrl::environments::cartpole::CartPole;
use myrl::train::Train;

#[test]
fn smoke_pg() {
    let mut env = CartPole::new();
    let mut agent =
        PGAgent::<CartPole>::new(env.state_dim(), env.action_space(), 0.99, tch::Device::Cpu);
    agent.train(&mut env, 100);
}

#[test]
fn smoke_a2c() {
    let mut env = CartPole::new();
    let mut agent =
        A2CAgent::<CartPole>::new(env.state_dim(), env.action_space(), 0.99, tch::Device::Cpu);
    agent.train(&mut env, 100);
}

#[test]
fn smoke_dqn() {
    let mut env = CartPole::new();
    let mut agent = DqnAgent::<CartPole>::new(
        env.state_dim(),
        env.action_space(),
        0.99,
        0.1,
        32,
        10,
        tch::Device::Cpu,
    );
    agent.train(&mut env, 100);
}

#[test]
fn smoke_trpo() {
    let mut env = CartPole::new();
    let mut agent = TrpoAgent::<CartPole>::new(
        env.state_dim(),
        env.action_space(),
        0.99,
        0.95,
        0.01,
        0.1,
        tch::Device::Cpu,
    );
    agent.train(&mut env, 5);
}
