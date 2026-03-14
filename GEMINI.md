# GEMINI.md - Project Context for MyRL

## Project Overview
**MyRL** is a modular Reinforcement Learning (RL) library written in Rust. It provides a clean set of traits and abstractions for implementing and training various RL algorithms using the **PyTorch (`tch-rs`)** backend for neural network computations.

### Key Technologies
- **Rust**: Language of implementation for safety and performance.
- **tch-rs**: Rust bindings for C++ PyTorch for deep learning.
- **ndarray**: Used for numerical computations (some parts might be transitioned or already in use).
- **plotters**: Used for generating training progress visualizations (e.g., `dqn_training.png`).

### Architecture
The project is organized into several key modules:
- `src/agent.rs`: Defines the `Agent` trait which abstracts the training loop.
- `src/environment.rs`: Defines the `Environment` trait, allowing the library to be agnostic of the specific task (e.g., CartPole).
- `src/algorithms/`: Contains implementations of RL algorithms:
    - **DQN** (Deep Q-Network)
    - **A2C** (Advantage Actor-Critic)
    - **Policy Gradient** (VPG - Vanilla Policy Gradient)
    - **TRPO** (Trust Region Policy Optimization)
- `src/environments/`: Contains environment implementations (e.g., `CartPole`).
- `src/networks/`: Contains neural network architectures like Multi-Layer Perceptrons (MLP).
- `src/policies/`: Contains action selection strategies (e.g., Epsilon-Greedy).

## Building and Running

### Prerequisites
- Rust (latest stable)
- Libtorch (required by `tch-rs`). Ensure `LIBTORCH` and `LD_LIBRARY_PATH` (or `DYLD_LIBRARY_PATH` on macOS) are correctly set.

### Commands
- **Build**: `cargo build`
- **Test**: `cargo test` (Runs integration tests for all implemented algorithms on CartPole).
- **Run Examples**: `cargo run --example <example_name>` (Check the `examples/` directory if it exists, otherwise use tests as a reference).

## Development Conventions

### Coding Style
- Follow standard Rust idiomatic practices (`cargo fmt`, `cargo clippy`).
- Traits like `Agent` and `Environment` are central; new algorithms or environments should implement these.
- Neural networks should be encapsulated within the `networks` module using `tch::nn`.

### Testing Practices
- Integration tests in `tests/` are used to verify algorithm convergence on standard environments like CartPole.
- Use `tch::Device::Cpu` for most tests unless GPU acceleration is explicitly needed and available.

### Contribution Guidelines
- When adding a new algorithm, place it in `src/algorithms/` and add an integration test in `tests/integration_test.rs`.
- Ensure training results can be visualized using the `plotters` crate by updating `src/utils.rs`.

## Key Files
- `src/lib.rs`: The entry point for the library, defining the module structure.
- `src/agent.rs`: The `Agent` trait definition.
- `src/environment.rs`: The `Environment` trait definition.
- `src/algorithms/dqn.rs`: Reference implementation for an off-policy algorithm.
- `src/algorithms/a2c.rs`: Reference implementation for an actor-critic algorithm.
- `src/environments/cartpole.rs`: The primary testbed environment.
