## Grid-based Resource Game and Trust-based Agent


The implementation lives in:

- `experiments/grid_trust_env.py` – environment definition
- `experiments/grid_trust_experiment.py` – training loop, agents, and logging

---

## 1. Environment: `GridTrustEnv`

`GridTrustEnv` is a small, MAgent-style grid world with shared resources and a simple governance rule that discourages over-farming without adding extra penalties.

### 1.1 Basic layout

Constructor signature (simplified):

```python
GridTrustEnv(
    grid_size: int = 10,
    num_agents: int = 4,
    num_resource_tiles: int = 12,
    resource_per_tile: int = 3,
    window_size: int = 8,
    harvest_threshold: int = 2,
    cooldown_steps: int = 30,
    max_steps: int = 200,
    seed: Optional[int] = 0,
)
```

- **`grid_size`**  
  Size of the square map. A value of `10` means a 10×10 grid. Larger values create more space for agents and resources.

- **`num_agents`**  
  Number of agents placed on the grid. All agents act in the same environment and compete for the same pool of resources.

- **`num_resource_tiles`**  
  Intended number of resource tiles. In the current implementation, the exact layout is controlled by the clustering logic (see below), so this field mainly encodes the desired resource scale.

- **`resource_per_tile`**  
  Initial amount of resource units on each resource tile.  
  A value of `3` means that a tile can grant up to three successful harvests before it is depleted.

- **`window_size`**  
  Time window (in environment steps) used to detect over-harvesting on a single tile.  
  The environment keeps a history of harvest times per tile and, at each step, counts how many harvests occurred in the last `window_size` steps.

- **`harvest_threshold`**  
  Maximum allowed number of harvests on a tile within the rolling window before it is considered over-exploited.  
  If the number of recent harvests exceeds this threshold, the tile and its local neighbourhood trigger a cooldown (see below).

- **`cooldown_steps`**  
  Duration of the cooldown, in steps, for tiles that are part of an over-harvested region.  
  During cooldown, harvesting from these tiles yields no reward. In addition, when cooldown is triggered, any remaining resources on the affected tiles are set to zero for the rest of the episode. This models long-term environmental damage without applying explicit negative rewards.

- **`max_steps`**  
  Maximum length of an episode in environment steps.  
  An episode terminates either when `max_steps` is reached or when all resources on the map have been depleted.

- **`seed`**  
  Random seed used to initialise the internal RNG. This makes the layout and dynamics reproducible.

### 1.2 Resource clusters and conflicts

The environment organises resources into spatial clusters:

- **`cluster_count`** (internal, default `2`)  
  Number of resource clusters. Each cluster is a small region containing several adjacent resource tiles.

- **`cluster_radius`** (internal, default `1`)  
  Radius of each resource cluster. A radius of `1` creates roughly a 3×3 block of resource tiles around each chosen cluster centre, clipped to the map boundaries.

The environment tracks additional statistics for analysis:

- **`cluster_map`**  
  Maps each resource coordinate to its cluster id.

- **`cluster_positions`**  
  Stores the list of positions for each resource cluster.

- **`cluster_depleted_step`**  
  Records the step at which each cluster becomes fully depleted. If a cluster survives until the end of the episode, its lifetime is treated as `max_steps`.

- **`conflict_events`**  
  Counts conflict events during an episode. A conflict occurs when more than one agent attempts to harvest from the same tile in the same step. Each extra harvester beyond the first contributes one conflict event.  
  This provides a simple measure of how often agents fight over exactly the same resource.

The method:

```python
get_episode_cluster_lifetimes() -> List[int]
```

returns the lifetimes of each cluster in steps, and:

```python
get_episode_conflict_level() -> float
```

returns the average number of extra harvesters per step (conflict intensity).

### 1.3 Actions and observations

- **Actions**  
  There are 6 discrete actions:

  ```text
  0: stay
  1: move up
  2: move down
  3: move left
  4: move right
  5: harvest
  ```

- **Observations**

  For each agent, `_get_obs()` returns a simple 4-dimensional vector:

  ```python
  [norm_x, norm_y, local_resource_sum, local_cooldown_count]
  ```

  - `norm_x, norm_y` – the agent’s current position normalised to `[0, 1]`, so that the state representation is independent of the absolute grid size.
  - `local_resource_sum` – total remaining resource in a 5×5 neighbourhood centred on the agent. This tells the agent whether it is in a rich or poor area.
  - `local_cooldown_count` – number of tiles in the same 5×5 window that are currently in cooldown. This provides a local signal of recent over-harvesting and sanctions.

---

## 2. Trust-based agent: `GridTrustAgent`

`GridTrustAgent` is a lightweight, trust-modulated Q-learning agent designed specifically for `GridTrustEnv`.

Key constructor:

```python
GridTrustAgent(
    obs_dim: int,
    n_actions: int,
    cfg: GridTrustConfig,
)
```

- **`obs_dim`**  
  Dimension of the observation vector. In this environment it is `4` (position, local resources, local cooldown).

- **`n_actions`**  
  Number of discrete actions. Here it is `6`.

- **`cfg`**  
  Training hyperparameters wrapped in `GridTrustConfig` (see below).

The agent maintains:

- **`weights`**  
  A `(n_actions, obs_dim)` weight matrix for linear Q-learning.  
  Each row corresponds to a linear value function over the observation for a specific action.

- **`success_events` and `sanction_events`**  
  Counters used to estimate trust.  
  - `success_events` counts how often the agent receives positive reward in reasonably healthy conditions.  
  - `sanction_events` counts how often the agent experiences zero or negative reward while many tiles are in cooldown, which approximates being “punished by the environment” for over-farming.

- **`trust_history`**  
  Log of the trust value over time, mainly for analysis and plotting.

### 2.1 Trust value

The current trust is computed as:

```python
def current_trust(self) -> float:
    total = self.success_events + self.sanction_events
    if total == 0:
        return 0.5
    return self.success_events / total
```

- If the agent has not yet experienced any clear success or sanction, its trust is initialised to `0.5` (neutral).
- As learning progresses, trust moves towards `1.0` if the agent often gains reward in healthy states, and towards `0.0` if it frequently receives no reward while the environment is heavily cooled down.

### 2.2 Action selection

The main decision function is:

```python
select_action(obs: np.ndarray, cooldown_tiles: float, training: bool = True) -> int
```

The logic is:

1. The observation is clipped for numerical stability.
2. The agent computes its current trust value.
3. If trust is low (`< 0.4`) and there is at least one cooldown tile on the map, it **forbids** the harvest action (action `5`) and biases the policy toward staying still or moving.  
   The idea is that when the environment appears “angry” (many cooldowns) and past behaviour has not gone well, the agent should stop greedily harvesting and let the system recover.
4. Otherwise, it uses epsilon-greedy exploration over the linear Q-values.

In short, trust acts as a brake on harvesting when the environment shows signs of over-exploitation.

### 2.3 Learning update and trust statistics

The update function:

```python
update(
    obs: np.ndarray,
    action: int,
    reward: float,
    next_obs: np.ndarray,
    done: bool,
    cooldown_tiles: float,
) -> None
```

performs two things:

1. **Linear Q-learning update**  
   - Computes the temporal-difference (TD) error between the current Q-value and a bootstrap target based on `gamma`.  
   - Updates the corresponding row in `weights` with learning rate `lr`.  
   - Clips both TD error and weights to avoid numerical issues.

2. **Trust statistics update**  
   - If the number of cooldown tiles is positive and the immediate reward is non-positive, the step is counted as a **sanction** event (with a slightly higher weight, `+2`).  
   - If the reward is strictly positive, the step is counted as a **success** event.  
   - After updating these counts, the new trust value is appended to `trust_history`.

This simple mechanism lets the agent infer how sustainable its behaviour is, purely from local observations and the global cooldown signal, without explicit communication or identity tracking.

---

## 3. Training configuration: `GridTrustConfig`

The training hyperparameters for both trust-based and baseline agents are grouped in:

```python
@dataclass
class GridTrustConfig:
    episodes: int = 2000
    max_steps: int = 500
    log_interval: int = 100
    gamma: float = 0.9
    lr: float = 0.01
    epsilon: float = 0.1
```

- **`episodes`**  
  Number of training episodes.

- **`max_steps`**  
  Maximum steps per episode. This value is also forwarded to the environment unless explicitly overridden in `env_kwargs`.

- **`log_interval`**  
  How often (in episodes) to print training statistics to the console.

- **`gamma`**  
  Discount factor for future rewards in the Q-learning update. Higher values place more weight on long-term outcomes.

- **`lr`**  
  Learning rate for the linear Q-learning updates.

- **`epsilon`**  
  Exploration probability in epsilon-greedy action selection.

---

## 4. Baseline agents and logged metrics

`GridTrustExperiment` also defines a baseline that uses the same linear Q-learning structure but **without** any trust shaping:

- Baseline agents keep their own weight matrices, follow epsilon-greedy Q-learning, and **do not** track trust.
- Their behaviour does not respond to cooldown signals beyond what is implicitly learned through rewards.

For both trust-based and baseline agents, the experiment logs several episode-level metrics:

- **Cooldown fraction** – average fraction of tiles in cooldown during an episode.  
  Lower values indicate less over-harvesting and a healthier resource pool.

- **Total remaining resources** – amount of resource left at the end of an episode.  
  Higher values indicate that agents have not drained the environment completely.

- **Reward Gini coefficient** – measures how unequal per-agent total rewards are.  
  Values near `0` indicate fair outcomes; values near `1` indicate that one or two agents captured almost all rewards.

- **Minimum return across agents** – captures how badly the worst-off agent performs.  
  Higher values suggest that cooperation or at least mutual restraint is protecting weaker agents.

- **Average cluster lifetime** – average time until each resource cluster is depleted.  
  Longer lifetimes mean that resource zones remain viable for more of the episode.

- **Conflict level** – average number of extra harvesters per step (from `get_episode_conflict_level`).  
  Lower conflict indicates that agents have learned to avoid simultaneous harvesting from the same tile.

These metrics together describe how the trust mechanism affects both individual performance and the overall health of the shared environment.


