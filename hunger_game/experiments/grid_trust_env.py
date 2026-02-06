import numpy as np
from typing import Dict, Tuple, List, Optional


class GridTrustEnv:
    """
    MAgent2-style grid environment with shared resource tiles and a
    rule-enforcing (cooldown) mechanism, implemented purely in Python.

    - Agents move on an N x N grid.
    - Some cells contain renewable resources.
    - Agents can harvest from the cell they occupy.
    - If a cell is over-harvested within a short time window, it enters
      a cooldown state where further harvest yields no reward.
    - No extra penalty reward is added; the only effect is that future
      positive rewards from that cell disappear during cooldown.
    """

    def __init__(
        self,
        grid_size: int = 10,
        num_agents: int = 4,
        num_resource_tiles: int = 12,
        resource_per_tile: int = 3,
        window_size: int = 8,
        harvest_threshold: int = 2,
        cooldown_steps: int = 30,
        max_steps: int = 200,
        seed: Optional[int] = 0,
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_resource_tiles = num_resource_tiles
        self.resource_per_tile = resource_per_tile
        self.window_size = window_size
        self.harvest_threshold = harvest_threshold
        self.cooldown_steps = cooldown_steps
        self.max_steps = max_steps

        self.rng = np.random.RandomState(seed)

        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]

        # cluster-based resource layout parameters
        # 2 moderately large clusters by default; can be overridden if needed
        self.cluster_count = 2
        self.cluster_radius = 1

        # cluster tracking
        self.cluster_map: Dict[Tuple[int, int], int] = {}
        self.cluster_positions: Dict[int, List[Tuple[int, int]]] = {}
        self.cluster_depleted_step: Dict[int, Optional[int]] = {}

        # conflict tracking
        self.conflict_events: int = 0
        self.steps_in_episode: int = 0

        # Actions: 0 stay, 1 up, 2 down, 3 left, 4 right, 5 harvest
        self.n_actions = 6

        # internal state
        self.step_count: int = 0
        self.agent_positions: Dict[str, np.ndarray] = {}
        self.resource_amounts: Dict[Tuple[int, int], int] = {}
        self.cooldown_remaining: Dict[Tuple[int, int], int] = {}
        self.harvest_history: Dict[Tuple[int, int], List[int]] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, np.ndarray]:
        self.step_count = 0

        # Randomize agent positions (allow overlap for simplicity)
        self.agent_positions = {
            aid: self.rng.randint(0, self.grid_size, size=2, dtype=int)
            for aid in self.agent_ids
        }

        # Clustered resource tiles: a few resource "zones" separated in space
        self.resource_amounts = {}
        self.cluster_map = {}
        self.cluster_positions = {}
        used = set()
        centers: List[Tuple[int, int]] = []
        attempts = 0
        # choose cluster centers that are reasonably far apart
        while len(centers) < self.cluster_count and attempts < 1000:
            attempts += 1
            cx = int(self.rng.randint(1, self.grid_size - 1))
            cy = int(self.rng.randint(1, self.grid_size - 1))
            # Manhattan distance >= 6 between cluster centers
            if all(abs(cx - ox) + abs(cy - oy) >= 6 for (ox, oy) in centers):
                centers.append((cx, cy))

        for cid, (cx, cy) in enumerate(centers):
            self.cluster_positions[cid] = []
            for dx in range(-self.cluster_radius, self.cluster_radius + 1):
                for dy in range(-self.cluster_radius, self.cluster_radius + 1):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        pos = (x, y)
                        if pos not in used:
                            self.resource_amounts[pos] = self.resource_per_tile
                            used.add(pos)
                        self.cluster_map[pos] = cid
                        self.cluster_positions[cid].append(pos)

        self.cooldown_remaining = {}
        self.harvest_history = {pos: [] for pos in self.resource_amounts.keys()}
        self.cluster_depleted_step = {cid: None for cid in self.cluster_positions}
        self.conflict_events = 0
        self.steps_in_episode = 0

        return self._get_obs()

    def step(self, actions: Dict[str, int]):
        """
        Args:
            actions: dict agent_id -> int in [0..5]
        Returns:
            obs: dict agent_id -> obs_vec
            rewards: dict agent_id -> float
            done: bool
            info: dict with logs
        """
        self.step_count += 1
        self.steps_in_episode += 1

        # 1) Move agents
        for aid, act in actions.items():
            if act in (0, 5):
                continue
            dx, dy = 0, 0
            if act == 1:
                dx = -1
            elif act == 2:
                dx = 1
            elif act == 3:
                dy = -1
            elif act == 4:
                dy = 1
            pos = self.agent_positions[aid]
            new_pos = pos + np.array([dx, dy], dtype=int)
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self.agent_positions[aid] = new_pos

        # 2) Decay cooldowns
        for pos in list(self.cooldown_remaining.keys()):
            self.cooldown_remaining[pos] -= 1
            if self.cooldown_remaining[pos] <= 0:
                del self.cooldown_remaining[pos]

        # 3) Harvest phase
        rewards: Dict[str, float] = {aid: 0.0 for aid in self.agent_ids}

        # track conflicts: multiple agents harvesting same cell
        pos_to_harvesters: Dict[Tuple[int, int], List[str]] = {}
        for aid, act in actions.items():
            if act == 5:
                pos = tuple(self.agent_positions[aid])
                pos_to_harvesters.setdefault(pos, []).append(aid)

        for pos, agents in pos_to_harvesters.items():
            if len(agents) > 1:
                # count conflicts as number of extra agents beyond the first
                self.conflict_events += len(agents) - 1

        for aid, act in actions.items():
            if act != 5:
                continue

            pos = tuple(self.agent_positions[aid])

            # if cell is in cooldown, nothing happens
            if pos in self.cooldown_remaining:
                continue

            # if no resource here, nothing
            if pos not in self.resource_amounts:
                continue

            if self.resource_amounts[pos] <= 0:
                continue

            # successful harvest
            self.resource_amounts[pos] -= 1
            rewards[aid] += 1.0
            self.harvest_history[pos].append(self.step_count)

            # check over-harvesting window
            recent = [
                t for t in self.harvest_history[pos] if self.step_count - t < self.window_size
            ]
            if len(recent) > self.harvest_threshold:
                # trigger cooldown on a 3x3 neighborhood around this tile
                x, y = pos
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < self.grid_size and 0 <= yy < self.grid_size:
                            p = (xx, yy)
                            self.cooldown_remaining[p] = self.cooldown_steps
                            # over-harvested area suffers long-term depletion:
                            # once a cell is in the cooled neighborhood, its
                            # remaining resources are set to zero for the rest
                            # of the episode (no negative reward, just no more food).
                            if p in self.resource_amounts:
                                self.resource_amounts[p] = 0

        # 4) termination: fixed horizon or all resources empty
        done = (
            self.step_count >= self.max_steps
            or all(v <= 0 for v in self.resource_amounts.values())
        )

        obs = self._get_obs()

        # global logs
        total_res = float(sum(self.resource_amounts.values()))
        cooldown_tiles = len(self.cooldown_remaining)
        info = {
            "total_resources": total_res,
            "cooldown_tiles": cooldown_tiles,
        }

        return obs, rewards, done, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        For each agent, return a vector:
        [norm_x, norm_y, local_resource_sum, local_cooldown_count]
        where local window is a 5x5 neighborhood around the agent.
        """
        obs: Dict[str, np.ndarray] = {}
        radius = 2
        for aid in self.agent_ids:
            x, y = self.agent_positions[aid]

            # normalized position
            nx = x / max(1, self.grid_size - 1)
            ny = y / max(1, self.grid_size - 1)

            # local window stats
            res_sum = 0.0
            cd_count = 0.0
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    xx = x + dx
                    yy = y + dy
                    if 0 <= xx < self.grid_size and 0 <= yy < self.grid_size:
                        pos = (xx, yy)
                        if pos in self.resource_amounts:
                            res_sum += float(self.resource_amounts[pos])
                        if pos in self.cooldown_remaining:
                            cd_count += 1.0

            vec = np.array([nx, ny, res_sum, cd_count], dtype=np.float32)
            obs[aid] = vec

        return obs

    # ------------------------------------------------------------------
    # Episode-level statistics helpers
    # ------------------------------------------------------------------
    def get_episode_cluster_lifetimes(self) -> List[int]:
        """Return per-cluster lifetimes (steps until depletion or max_steps)."""
        lifetimes: List[int] = []
        for cid in self.cluster_positions:
            step = self.cluster_depleted_step.get(cid)
            if step is None:
                lifetimes.append(self.max_steps)
            else:
                lifetimes.append(step)
        return lifetimes

    def get_episode_conflict_level(self) -> float:
        """Average number of extra harvesters per step (conflict intensity)."""
        if self.steps_in_episode == 0:
            return 0.0
        return float(self.conflict_events) / float(self.steps_in_episode)


if __name__ == "__main__":
    env = GridTrustEnv()
    obs = env.reset()
    done = False
    total_steps = 0
    while not done:
        actions = {aid: 5 for aid in env.agent_ids}  # all harvest
        obs, rewards, done, info = env.step(actions)
        total_steps += 1
    print(f"Ran {total_steps} steps, remaining resources={info['total_resources']}")


