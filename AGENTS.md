# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the library crate. Core traits live in `src/agent.rs`, `src/environment.rs`, and `src/policy.rs`. Algorithm implementations are grouped under `src/algorithms/` (`dqn.rs`, `a2c.rs`, `policy_gradient.rs`, `TRPO.rs`), reusable models under `src/networks/`, and policies under `src/policies/`. Built-in environments are in `src/environments/`, currently `cartpole.rs`. Cross-cutting helpers such as plotting and tensor conversion live in `src/utils.rs`. Integration coverage sits in `tests/integration_test.rs`. Training plots and other reference images are stored in `assets/`.

## Build, Test, and Development Commands
Use Cargo for the standard workflow:

- `cargo build`: compile the library and tests.
- `cargo test`: run the integration suite against CartPole.
- `cargo test dqn_cart_pole_test -- --nocapture`: run one long-form training test and keep episode logs visible.
- `cargo fmt`: format Rust sources before opening a PR.
- `cargo clippy --all-targets --all-features`: catch common Rust issues before review.
- `fish run_debug.fish`: build tests and open the integration binary in `lldb` on macOS.

## Coding Style & Naming Conventions
Follow default Rust style: 4-space indentation, `snake_case` for functions/modules, `CamelCase` for structs/traits, and short, explicit generic bounds. Keep new modules lowercase when possible; `src/algorithms/TRPO.rs` is an existing exception, not a pattern to copy. Prefer small trait-focused modules over large monolithic files. Run `cargo fmt` after edits and keep public APIs in `src/lib.rs` aligned with the module tree.

## Testing Guidelines
This repo currently relies on integration tests in [`tests/integration_test.rs`](/Users/baoyihui/Documents/coding/vscode/rust_space/myrl/tests/integration_test.rs). Name new tests after the algorithm and environment they exercise, for example `ppo_cart_pole_test`. Keep default episode counts practical for local runs, and avoid unnecessary plotting in tests unless the output is being inspected. If a change touches environment dynamics or tensor shapes, add or extend an integration test.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit prefixes such as `feat:`. Continue with `feat:`, `fix:`, `refactor:`, or `test:` followed by a concise summary. PRs should describe the behavioral change, list validation commands run, and attach updated plots or screenshots when training output changes.

## Environment & Configuration Notes
The crate depends on `tch`, so local runs may require a working libtorch/PyTorch setup. The existing `run_debug.fish` script expects a Conda environment named `torch_env` and sets `DYLD_LIBRARY_PATH` for macOS debugging.
