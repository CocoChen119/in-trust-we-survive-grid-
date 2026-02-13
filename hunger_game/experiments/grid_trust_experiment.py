import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .grid_trust_env import GridTrustEnv


@dataclass
class GridTrustConfig:
    episodes: int = 2000
    max_steps: int = 500
    log_interval: int = 100
    gamma: float = 0.9
    lr: float = 0.01
    epsilon: float = 0.1


class GridTrustAgent:
    """
    Trust-based agent for the custom grid environment.

    - Linear Q-learning over a small state vector [x, y, local_res, local_cd]
    - Maintains a trust score based on balance of "clean" rewards vs.
      interactions during high-cooldown phases (approximation of sanctions).
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: GridTrustConfig):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg

        self.weights = 0.01 * np.random.randn(n_actions, obs_dim).astype(np.float32)
        self.success_events = 0
        self.sanction_events = 0
        self.trust_history: list[float] = []

    def current_trust(self) -> float:
        total = self.success_events + self.sanction_events
        if total == 0:
            return 0.5
        return self.success_events / total

    def select_action(self, obs: np.ndarray, cooldown_tiles: float, training: bool = True) -> int:
        # normalize & clip
        s = np.clip(obs, -5.0, 5.0)

        # when trust is low or many cooldown tiles, forbid harvest (action 5)
        # and strongly prefer stay / move actions.
        trust = self.current_trust()
        forbid_harvest = trust < 0.4 and cooldown_tiles > 0

        if training and np.random.rand() < self.cfg.epsilon:
            if forbid_harvest:
                return int(np.random.randint(0, self.n_actions - 1))  # 0..4
            return int(np.random.randint(self.n_actions))

        q_values = self.weights @ s
        if forbid_harvest:
            q_values[5] = -1e9  # effectively remove harvest action
            q_values[0] += 0.3  # stronger bias toward "stay"

        return int(np.argmax(q_values))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        cooldown_tiles: float,
    ) -> None:
        s = np.clip(obs, -5.0, 5.0)
        s_next = np.clip(next_obs, -5.0, 5.0)

        q_curr = float(self.weights[action] @ s)
        q_next = float(np.max(self.weights @ s_next))
        target = reward + (0.0 if done else self.cfg.gamma * q_next)
        td_error = target - q_curr

        td_error = float(np.clip(td_error, -10.0, 10.0))
        self.weights[action] += self.cfg.lr * td_error * s
        self.weights = np.clip(self.weights, -50.0, 50.0)

        # trust statistics: if global cooldown high and reward <= 0, treat as sanction
        if cooldown_tiles > 0 and reward <= 0.0:
            # stronger negative update when still trying to harvest under high cooldown
            self.sanction_events += 2
        elif reward > 0.0:
            self.success_events += 1

        self.trust_history.append(self.current_trust())


class GridTrustExperiment:
    def __init__(
        self,
        cfg: Optional[GridTrustConfig] = None,
        env_kwargs: Optional[dict] = None,
        save_suffix: str = "",
    ):
        # fix random seeds for reproducibility of figures
        np.random.seed(0)
        random.seed(0)

        self.cfg = cfg or GridTrustConfig()
        if env_kwargs is None:
            env_kwargs = {}
        # ensure horizon matches config unless explicitly overridden
        env_kwargs.setdefault("max_steps", self.cfg.max_steps)
        self.env = GridTrustEnv(**env_kwargs)
        self.save_suffix = save_suffix

        obs0 = self.env.reset()
        example_agent = self.env.agent_ids[0]
        obs_dim = obs0[example_agent].shape[0]
        n_actions = self.env.n_actions

        # trust-based agents
        self.trust_agents: Dict[str, GridTrustAgent] = {
            aid: GridTrustAgent(obs_dim, n_actions, self.cfg)
            for aid in self.env.agent_ids
        }

        # simple baseline agents: same linear Q-learning but without trust shaping
        self.base_weights: Dict[str, np.ndarray] = {
            aid: 0.01 * np.random.randn(n_actions, obs_dim).astype(np.float32)
            for aid in self.env.agent_ids
        }

        # logs for trust experiment
        self.trust_episode_rewards: Dict[str, list[float]] = {
            aid: [] for aid in self.env.agent_ids
        }
        self.trust_episode_trust: Dict[str, list[float]] = {
            aid: [] for aid in self.env.agent_ids
        }
        self.trust_cooldown_fraction: list[float] = []
        self.trust_total_resources: list[float] = []
        self.trust_gini_rewards: list[float] = []

        # cooperation-related metrics (trust) – used for analysis, not all plotted
        self.trust_min_return: list[float] = []          # worst-agent return
        self.trust_cluster_lifetime: list[float] = []    # avg cluster lifetime
        self.trust_conflict_level: list[float] = []      # avg conflict per step

        # logs for baseline experiment
        self.base_episode_rewards: Dict[str, list[float]] = {
            aid: [] for aid in self.env.agent_ids
        }
        self.base_cooldown_fraction: list[float] = []
        self.base_total_resources: list[float] = []
        self.base_gini_rewards: list[float] = []

        # cooperation-related metrics (baseline)
        self.base_min_return: list[float] = []
        self.base_cluster_lifetime: list[float] = []
        self.base_conflict_level: list[float] = []

    def run(self) -> None:
        # 1) run trust-based experiment
        for ep in range(self.cfg.episodes):
            obs = self.env.reset()
            done = False
            ep_rew = {aid: 0.0 for aid in self.env.agent_ids}
            cd_per_step: list[float] = []
            res_per_step: list[float] = []

            while not done:
                cd_tiles = float(len(self.env.cooldown_remaining))
                total_tiles = float(self.env.grid_size * self.env.grid_size)
                cd_frac = cd_tiles / max(1.0, total_tiles)

                actions = {}
                for aid in self.env.agent_ids:
                    actions[aid] = self.trust_agents[aid].select_action(
                        obs[aid], cooldown_tiles=cd_tiles, training=True
                    )

                next_obs, rewards, done, info = self.env.step(actions)

                for aid in self.env.agent_ids:
                    self.trust_agents[aid].update(
                        obs[aid],
                        actions[aid],
                        rewards[aid],
                        next_obs[aid],
                        done,
                        cooldown_tiles=cd_tiles,
                    )
                    ep_rew[aid] += rewards[aid]

                obs = next_obs
                cd_per_step.append(cd_frac)
                res_per_step.append(info["total_resources"])

            self.trust_cooldown_fraction.append(
                float(np.mean(cd_per_step)) if cd_per_step else 0.0
            )
            self.trust_total_resources.append(
                float(res_per_step[-1]) if res_per_step else 0.0
            )

            # per-episode reward fairness (trust)
            rewards_vec = np.array(
                [ep_rew[aid] for aid in self.env.agent_ids], dtype=np.float32
            )
            self.trust_gini_rewards.append(self._gini(rewards_vec))

            # per-episode worst-case return (trust)
            self.trust_min_return.append(float(np.min(rewards_vec)))

            # per-episode cluster lifetime and conflict level
            lifetimes = self.env.get_episode_cluster_lifetimes()
            self.trust_cluster_lifetime.append(float(np.mean(lifetimes)))
            self.trust_conflict_level.append(self.env.get_episode_conflict_level())

            for aid in self.env.agent_ids:
                self.trust_episode_rewards[aid].append(ep_rew[aid])
                th = self.trust_agents[aid].trust_history
                self.trust_episode_trust[aid].append(th[-1] if th else 0.5)

            if (ep + 1) % self.cfg.log_interval == 0:
                avg_r = np.mean(
                    [
                        self.trust_episode_rewards[aid][-self.cfg.log_interval :]
                        for aid in self.env.agent_ids
                    ]
                )
                avg_trust = np.mean(
                    [
                        self.trust_episode_trust[aid][-self.cfg.log_interval :]
                        for aid in self.env.agent_ids
                    ]
                )
                avg_cd = np.mean(self.trust_cooldown_fraction[-self.cfg.log_interval :])
                print(
                    f"[TRUST] Episode {ep + 1}/{self.cfg.episodes} | "
                    f"avg reward={avg_r:.3f} | avg trust={avg_trust:.3f} | "
                    f"cooldown fraction={avg_cd:.3f}"
                )

        # 2) run baseline experiment (no trust shaping)
        for ep in range(self.cfg.episodes):
            obs = self.env.reset()
            done = False
            ep_rew = {aid: 0.0 for aid in self.env.agent_ids}
            cd_per_step: list[float] = []
            res_per_step: list[float] = []

            while not done:
                cd_tiles = float(len(self.env.cooldown_remaining))
                total_tiles = float(self.env.grid_size * self.env.grid_size)
                cd_frac = cd_tiles / max(1.0, total_tiles)

                actions = {}
                for aid in self.env.agent_ids:
                    # simple epsilon-greedy over linear Q, without trust bias
                    if np.random.rand() < self.cfg.epsilon:
                        actions[aid] = np.random.randint(self.env.n_actions)
                    else:
                        w = self.base_weights[aid]
                        s = np.clip(obs[aid], -5.0, 5.0)
                        q_vals = w @ s
                        actions[aid] = int(np.argmax(q_vals))

                next_obs, rewards, done, info = self.env.step(actions)

                for aid in self.env.agent_ids:
                    w = self.base_weights[aid]
                    s = np.clip(obs[aid], -5.0, 5.0)
                    s_next = np.clip(next_obs[aid], -5.0, 5.0)
                    q_curr = float(w[actions[aid]] @ s)
                    q_next = float(np.max(w @ s_next))
                    target = rewards[aid] + (0.0 if done else self.cfg.gamma * q_next)
                    td_error = float(np.clip(target - q_curr, -10.0, 10.0))
                    w[actions[aid]] += self.cfg.lr * td_error * s
                    self.base_weights[aid] = np.clip(w, -50.0, 50.0)
                    ep_rew[aid] += rewards[aid]

                obs = next_obs
                cd_per_step.append(cd_frac)
                res_per_step.append(info["total_resources"])

            self.base_cooldown_fraction.append(
                float(np.mean(cd_per_step)) if cd_per_step else 0.0
            )
            self.base_total_resources.append(
                float(res_per_step[-1]) if res_per_step else 0.0
            )

            for aid in self.env.agent_ids:
                self.base_episode_rewards[aid].append(ep_rew[aid])

            # per-episode reward fairness (baseline)
            rewards_vec = np.array(
                [ep_rew[aid] for aid in self.env.agent_ids], dtype=np.float32
            )
            self.base_gini_rewards.append(self._gini(rewards_vec))

            # per-episode worst-case return (baseline)
            self.base_min_return.append(float(np.min(rewards_vec)))

            # per-episode cluster lifetime and conflict level
            lifetimes = self.env.get_episode_cluster_lifetimes()
            self.base_cluster_lifetime.append(float(np.mean(lifetimes)))
            self.base_conflict_level.append(self.env.get_episode_conflict_level())

            if (ep + 1) % self.cfg.log_interval == 0:
                avg_r = np.mean(
                    [
                        self.base_episode_rewards[aid][-self.cfg.log_interval :]
                        for aid in self.env.agent_ids
                    ]
                )
                avg_cd = np.mean(self.base_cooldown_fraction[-self.cfg.log_interval :])
                print(
                    f"[BASE] Episode {ep + 1}/{self.cfg.episodes} | "
                    f"avg reward={avg_r:.3f} | cooldown fraction={avg_cd:.3f}"
                )

        # Print a short numerical summary to the console for analysis
        self._print_summary()
        # Main figure: cooldown + conflict in one image for easy comparison
        self._plot_results()

    # ---------------- plotting ----------------
    def _smooth(self, arr: list[float], window: int = 80) -> np.ndarray:
        # default: use window=80 for main plots, override explicitly if needed
        if window is None:
            window = 80
        if len(arr) == 0 or len(arr) < window or window <= 1:
            return np.asarray(arr, dtype=np.float32)
        out = []
        for i in range(len(arr)):
            s = max(0, i - window + 1)
            out.append(np.mean(arr[s : i + 1]))
        return np.asarray(out, dtype=np.float32)

    def _gini(self, x: np.ndarray) -> float:
        """Compute Gini coefficient of a 1D array (0=fair, 1=very unequal)."""
        x = np.asarray(x, dtype=np.float64).flatten()
        if x.size == 0:
            return 0.0
        # shift to non-negative
        x = x - x.min()
        if np.allclose(x, 0):
            return 0.0
        x = np.sort(x)
        n = x.size
        cumx = np.cumsum(x)
        g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
        return float(max(0.0, min(1.0, g)))

    def _print_summary(self) -> None:
        """Print scalar metrics that are not necessarily plotted."""
        def avg(xs: list[float]) -> float:
            return float(np.mean(xs[-200:])) if xs else 0.0

        print("\n=== Summary over last 200 episodes ===")
        print(
            f"Cooldown (trust vs base): "
            f"{avg(self.trust_cooldown_fraction):.4f} vs {avg(self.base_cooldown_fraction):.4f}"
        )
        print(
            f"Conflict (trust vs base): "
            f"{avg(self.trust_conflict_level):.4f} vs {avg(self.base_conflict_level):.4f}"
        )
        print(
            f"Cluster lifetime (trust vs base): "
            f"{avg(self.trust_cluster_lifetime):.1f} vs {avg(self.base_cluster_lifetime):.1f}"
        )
        print(
            f"Min return (trust vs base): "
            f"{avg(self.trust_min_return):.3f} vs {avg(self.base_min_return):.3f}"
        )

        # simple cooperation index: fraction of episodes where all agents get
        # non-trivial positive reward
        def coop_index(min_returns: list[float], threshold: float = 0.1) -> float:
            if not min_returns:
                return 0.0
            vals = np.array(min_returns, dtype=np.float32)
            return float(np.mean(vals[-200:] > threshold))

        ci_trust = coop_index(self.trust_min_return)
        ci_base = coop_index(self.base_min_return)
        print(
            f"Cooperation index (all agents >= 0.1 reward): "
            f"{ci_trust:.3f} (trust) vs {ci_base:.3f} (baseline)"
        )

    def _plot_results(self) -> None:
        base_dir = os.path.join(os.path.dirname(__file__), "..")
        save_path = os.path.join(
            base_dir,
            f"grid_trust_results{self.save_suffix}.png",
        )

        # 1x2 grid: cooldown + conflict
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 1) cooldown fraction comparison
        ax = axes[0]
        sm_cd_trust = self._smooth(self.trust_cooldown_fraction, window=80)
        sm_cd_base = self._smooth(self.base_cooldown_fraction, window=80)
        ax.plot(sm_cd_trust, color="tab:red", linewidth=1.8, label="Trust-based")
        ax.plot(
            sm_cd_base,
            color="tab:orange",
            linewidth=1.8,
            linestyle="--",
            label="Baseline",
        )
        ax.set_title("Average cooldown fraction (lower is better)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cooldown fraction")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # 2) conflict level comparison (use stronger smoothing)
        ax = axes[1]
        sm_conf_trust = self._smooth(self.trust_conflict_level, window=150)
        sm_conf_base = self._smooth(self.base_conflict_level, window=150)
        ax.plot(
            sm_conf_trust,
            color="tab:green",
            linewidth=1.8,
            label="Conflict (trust)",
        )
        ax.plot(
            sm_conf_base,
            color="tab:blue",
            linewidth=1.8,
            linestyle="--",
            label="Conflict (baseline)",
        )
        ax.set_title("Average conflict level per episode (lower is better)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg conflicts per step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # additional figure: learning curves (episode rewards over training)
        learn_path = os.path.join(
            base_dir,
            f"grid_learning_curves{self.save_suffix}.png",
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        # average episode reward (trust vs baseline)
        # per-episode mean over agents
        trust_ep_mean = []
        base_ep_mean = []
        num_eps = len(next(iter(self.trust_episode_rewards.values()), []))
        for ep in range(num_eps):
            trust_vals = [
                self.trust_episode_rewards[aid][ep]
                for aid in self.env.agent_ids
            ]
            base_vals = [
                self.base_episode_rewards[aid][ep]
                for aid in self.env.agent_ids
            ]
            trust_ep_mean.append(float(np.mean(trust_vals)))
            base_ep_mean.append(float(np.mean(base_vals)))

        sm_trust_rew = self._smooth(trust_ep_mean, window=80)
        sm_base_rew = self._smooth(base_ep_mean, window=80)
        ax.plot(sm_trust_rew, color="tab:red", linewidth=1.8, label="Trust-based")
        ax.plot(
            sm_base_rew,
            color="tab:blue",
            linewidth=1.8,
            linestyle="--",
            label="Baseline",
        )
        ax.set_title("Average episode reward over training")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(learn_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    # Main configuration used in the paper (medium difficulty, social dilemma present)
    cfg = GridTrustConfig(episodes=1500, max_steps=500, log_interval=100)
    env_kwargs = {
        "resource_per_tile": 3,
        "window_size": 8,
        "harvest_threshold": 2,
        "cooldown_steps": 30,
    }

    print("\n=== Running grid trust experiment (main configuration) ===")
    exp = GridTrustExperiment(cfg, env_kwargs=env_kwargs, save_suffix="")
    exp.run()


