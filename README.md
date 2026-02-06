## Hunger Game Project

A multi-agent reinforcement learning simulation where agents learn to survive in a tower environment by managing resources and making strategic decisions about food consumption.

### Project Structure
`hunger_game/`
- `config/config.json`
- `src/`
  - `agents/` (`q_learning.py`, `monte_carlo.py`, `qianpu.py`, `base_agent.py`)
  - `environment.py`
  - `training.py`
  - `visualization.py`
  - `metrics.py`
- `experiments/`
  - `grid_trust_env.py`
  - `grid_trust_experiment.py`
  - `iegs_variants.py`
- `main.py`

### Features
- **Tower environment**: multi-agent survival with resource sharing and hunger dynamics.
- **Algorithms**: Q-learning, Monte Carlo, and an imitation/evolution-style trust agent (IE‑GS).
- **Visualization**: Pygame-based visualization of the tower scenario.
- **Metrics & plots**: training curves, early-phase analysis, and algorithm comparison.
- **Configurable**: environment and training settings via `config/config.json`.

### Installation

```bash
git clone https://github.com/your-username/hunger-game.git
cd hunger-game
pip install -r requirements.txt
```

### Running the tower experiments

```bash
cd hunger_game
python main.py
```

This trains Q-learning, Monte Carlo, and IE‑GS agents in the tower environment and writes result figures (learning curves and comparisons) into the project root.

### Grid trust experiment (cooldown rule environment)

The `experiments/grid_trust_env.py` and `experiments/grid_trust_experiment.py` modules implement a separate MAgent2-style grid world with **soft rule enforcement**:
- Agents move on a 2D grid with clustered resource tiles.
- Over-harvesting a local resource cluster triggers a **cooldown** where tiles stop yielding positive reward (no explicit penalties).
- We compare a **trust-based agent** against a plain Q-learning baseline on metrics like cooldown fraction and conflict level.

To reproduce the grid trust results figure:

```bash
cd hunger_game
python -m experiments.grid_trust_experiment
```

This command generates `grid_trust_results.png`, containing:
- Left: average cooldown fraction over episodes (trust vs baseline).
- Right: average conflict level (simultaneous harvests on the same cell) over episodes.

### License

[MIT](https://choosealicense.com/licenses/mit/)
