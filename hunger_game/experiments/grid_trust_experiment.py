import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .grid_trust_env import GridTrustEnv

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None


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
        global_seed: Optional[int] = 0,
    ):
        # fix random seeds for reproducibility / per-run variability
        if global_seed is not None:
            np.random.seed(global_seed)
            random.seed(global_seed)

        self.cfg = cfg or GridTrustConfig()
        if env_kwargs is None:
            env_kwargs = {}
        # ensure horizon matches config unless explicitly overridden
        env_kwargs.setdefault("max_steps", self.cfg.max_steps)
        self.env_kwargs = dict(env_kwargs)
        self.env = GridTrustEnv(**self.env_kwargs)
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
        # pre-generate episode seeds so that trust and baseline see
        # the same sequence of environment realisations
        episode_seeds = [
            int(np.random.randint(0, 10**9)) for _ in range(self.cfg.episodes)
        ]

        # 1) run trust-based experiment
        for ep, ep_seed in enumerate(episode_seeds):
            env = GridTrustEnv(seed=ep_seed, **self.env_kwargs)
            obs = env.reset()
            done = False
            ep_rew = {aid: 0.0 for aid in env.agent_ids}
            cd_per_step: list[float] = []
            res_per_step: list[float] = []

            while not done:
                cd_tiles = float(len(env.cooldown_remaining))
                total_tiles = float(env.grid_size * env.grid_size)
                cd_frac = cd_tiles / max(1.0, total_tiles)

                actions = {}
                for aid in env.agent_ids:
                    actions[aid] = self.trust_agents[aid].select_action(
                        obs[aid], cooldown_tiles=cd_tiles, training=True
                    )

                next_obs, rewards, done, info = env.step(actions)

                for aid in env.agent_ids:
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
                [ep_rew[aid] for aid in env.agent_ids], dtype=np.float32
            )
            self.trust_gini_rewards.append(self._gini(rewards_vec))

            # per-episode worst-case return (trust)
            self.trust_min_return.append(float(np.min(rewards_vec)))

            # per-episode cluster lifetime and conflict level
            lifetimes = env.get_episode_cluster_lifetimes()
            self.trust_cluster_lifetime.append(float(np.mean(lifetimes)))
            self.trust_conflict_level.append(env.get_episode_conflict_level())

            for aid in env.agent_ids:
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
        for ep, ep_seed in enumerate(episode_seeds):
            env = GridTrustEnv(seed=ep_seed, **self.env_kwargs)
            obs = env.reset()
            done = False
            ep_rew = {aid: 0.0 for aid in env.agent_ids}
            cd_per_step: list[float] = []
            res_per_step: list[float] = []

            while not done:
                cd_tiles = float(len(env.cooldown_remaining))
                total_tiles = float(env.grid_size * env.grid_size)
                cd_frac = cd_tiles / max(1.0, total_tiles)

                actions = {}
                for aid in env.agent_ids:
                    # simple epsilon-greedy over linear Q, without trust bias
                    if np.random.rand() < self.cfg.epsilon:
                        actions[aid] = np.random.randint(env.n_actions)
                    else:
                        w = self.base_weights[aid]
                        s = np.clip(obs[aid], -5.0, 5.0)
                        q_vals = w @ s
                        actions[aid] = int(np.argmax(q_vals))

                next_obs, rewards, done, info = env.step(actions)

                for aid in env.agent_ids:
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

            for aid in env.agent_ids:
                self.base_episode_rewards[aid].append(ep_rew[aid])

            # per-episode reward fairness (baseline)
            rewards_vec = np.array(
                [ep_rew[aid] for aid in self.env.agent_ids], dtype=np.float32
            )
            self.base_gini_rewards.append(self._gini(rewards_vec))

            # per-episode worst-case return (baseline)
            self.base_min_return.append(float(np.min(rewards_vec)))

            # per-episode cluster lifetime and conflict level (baseline)
            lifetimes = env.get_episode_cluster_lifetimes()
            self.base_cluster_lifetime.append(float(np.mean(lifetimes)))
            self.base_conflict_level.append(env.get_episode_conflict_level())

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
        # Save per-episode statistics to disk for further analysis
        self._save_stats()

    def record_trust_episode_gif(
        self,
        out_path: str = "grid_trust_episode.gif",
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Run a single episode with the trained trust-based agents and
        record a simple GIF animation of the grid.

        This requires imageio to be installed. You can install it with:
            pip install imageio
        """
        if imageio is None:
            raise RuntimeError(
                "imageio is not available. Please install it with 'pip install imageio' "
                "to enable GIF recording."
            )

        obs = self.env.reset()
        done = False
        frames: list[np.ndarray] = []
        step_limit = max_steps if max_steps is not None else self.cfg.max_steps

        step = 0
        while not done and step < step_limit:
            cd_tiles = float(len(self.env.cooldown_remaining))
            actions = {}
            for aid in self.env.agent_ids:
                actions[aid] = self.trust_agents[aid].select_action(
                    obs[aid], cooldown_tiles=cd_tiles, training=False
                )

            obs, rewards, done, info = self.env.step(actions)
            frame = self.env.render(mode="rgb_array")
            frames.append(frame)
            step += 1

        imageio.mimsave(out_path, frames, fps=3)

    def record_baseline_episode_gif(
        self,
        out_path: str = "grid_baseline_episode.gif",
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Run a single episode with the baseline agents (no trust shaping)
        and record a GIF animation.
        """
        if imageio is None:
            raise RuntimeError(
                "imageio is not available. Please install it with 'pip install imageio' "
                "to enable GIF recording."
            )

        # fresh env with same configuration
        self.env = GridTrustEnv(**self.env_kwargs)
        obs = self.env.reset()
        done = False
        frames: list[np.ndarray] = []
        step_limit = max_steps if max_steps is not None else self.cfg.max_steps

        step = 0
        while not done and step < step_limit:
            cd_tiles = float(len(self.env.cooldown_remaining))
            actions = {}
            for aid in self.env.agent_ids:
                # greedy w.r.t. learned weights (no epsilon) for clearer behaviour
                w = self.base_weights[aid]
                s = np.clip(obs[aid], -5.0, 5.0)
                q_vals = w @ s
                actions[aid] = int(np.argmax(q_vals))

            obs, rewards, done, info = self.env.step(actions)
            frame = self.env.render(mode="rgb_array")
            frames.append(frame)
            step += 1

        imageio.mimsave(out_path, frames, fps=3)

    def record_comparison_gif(
        self,
        out_path: str = "grid_trust_vs_baseline.gif",
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Record a side-by-side GIF comparing a trust-based episode (left)
        and a baseline episode (right).
        """
        if imageio is None:
            raise RuntimeError(
                "imageio is not available. Please install it with 'pip install imageio' "
                "to enable GIF recording."
            )

        # 1) trust-based episode frames
        obs = self.env.reset()
        done = False
        trust_frames: list[np.ndarray] = []
        step_limit = max_steps if max_steps is not None else self.cfg.max_steps
        step = 0
        while not done and step < step_limit:
            cd_tiles = float(len(self.env.cooldown_remaining))
            actions = {}
            for aid in self.env.agent_ids:
                actions[aid] = self.trust_agents[aid].select_action(
                    obs[aid], cooldown_tiles=cd_tiles, training=False
                )
            obs, rewards, done, info = self.env.step(actions)
            trust_frames.append(self.env.render(mode="rgb_array"))
            step += 1

        # 2) baseline episode frames (fresh env)
        base_env = GridTrustEnv(**self.env_kwargs)
        obs = base_env.reset()
        done = False
        base_frames: list[np.ndarray] = []
        step = 0
        while not done and step < step_limit:
            cd_tiles = float(len(base_env.cooldown_remaining))
            actions = {}
            for aid in base_env.agent_ids:
                w = self.base_weights[aid]
                s = np.clip(obs[aid], -5.0, 5.0)
                q_vals = w @ s
                actions[aid] = int(np.argmax(q_vals))
            obs, rewards, done, info = base_env.step(actions)
            base_frames.append(base_env.render(mode="rgb_array"))
            step += 1

        # 3) align lengths and stack horizontally
        n = min(len(trust_frames), len(base_frames))
        combined_frames: list[np.ndarray] = []
        for i in range(n):
            left = trust_frames[i]
            right = base_frames[i]
            # ensure same height
            h = min(left.shape[0], right.shape[0])
            left = left[:h, :, :]
            right = right[:h, :, :]
            combined = np.concatenate([left, right], axis=1)
            combined_frames.append(combined)

        print(f"[GIF] trust frames={len(trust_frames)}, baseline frames={len(base_frames)}, combined={len(combined_frames)}")
        imageio.mimsave(out_path, combined_frames, fps=3)

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

    def _save_stats(self) -> None:
        """
        Save per-episode statistics for trust-based and baseline runs to CSV.

        This makes it easy to analyse results without scraping console output.
        """
        base_dir = os.path.join(os.path.dirname(__file__), "..")
        csv_path = os.path.join(base_dir, f"grid_stats{self.save_suffix}.csv")

        num_eps = len(self.trust_cooldown_fraction)
        # basic safety check
        if num_eps == 0:
            return

        import csv

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "trust_cooldown",
                    "base_cooldown",
                    "trust_conflict",
                    "base_conflict",
                    "trust_min_return",
                    "base_min_return",
                    "trust_total_resources",
                    "base_total_resources",
                    "trust_gini_rewards",
                    "base_gini_rewards",
                ]
            )
            for ep in range(num_eps):
                writer.writerow(
                    [
                        ep + 1,
                        self.trust_cooldown_fraction[ep]
                        if ep < len(self.trust_cooldown_fraction)
                        else "",
                        self.base_cooldown_fraction[ep]
                        if ep < len(self.base_cooldown_fraction)
                        else "",
                        self.trust_conflict_level[ep]
                        if ep < len(self.trust_conflict_level)
                        else "",
                        self.base_conflict_level[ep]
                        if ep < len(self.base_conflict_level)
                        else "",
                        self.trust_min_return[ep]
                        if ep < len(self.trust_min_return)
                        else "",
                        self.base_min_return[ep]
                        if ep < len(self.base_min_return)
                        else "",
                        self.trust_total_resources[ep]
                        if ep < len(self.trust_total_resources)
                        else "",
                        self.base_total_resources[ep]
                        if ep < len(self.base_total_resources)
                        else "",
                        self.trust_gini_rewards[ep]
                        if ep < len(self.trust_gini_rewards)
                        else "",
                        self.base_gini_rewards[ep]
                        if ep < len(self.base_gini_rewards)
                        else "",
                    ]
                )

    def summary_metrics(self, last_k: int = 200) -> dict:
        """
        Compute scalar summary metrics over the last `last_k` episodes.

        This is useful both for printing a per-run summary and for aggregating
        statistics across multiple random seeds.
        """

        def avg(xs: list[float]) -> float:
            if not xs:
                return 0.0
            xs_arr = np.asarray(xs[-last_k:], dtype=np.float32)
            return float(xs_arr.mean())

        # simple cooperation index: fraction of episodes where all agents get
        # non-trivial positive reward
        def coop_index(min_returns: list[float], threshold: float = 0.1) -> float:
            if not min_returns:
                return 0.0
            vals = np.array(min_returns[-last_k:], dtype=np.float32)
            return float(np.mean(vals > threshold))

        metrics = {
            "cooldown_trust": avg(self.trust_cooldown_fraction),
            "cooldown_base": avg(self.base_cooldown_fraction),
            "conflict_trust": avg(self.trust_conflict_level),
            "conflict_base": avg(self.base_conflict_level),
            "cluster_lifetime_trust": avg(self.trust_cluster_lifetime),
            "cluster_lifetime_base": avg(self.base_cluster_lifetime),
            "min_return_trust": avg(self.trust_min_return),
            "min_return_base": avg(self.base_min_return),
            "coop_index_trust": coop_index(self.trust_min_return),
            "coop_index_base": coop_index(self.base_min_return),
        }
        return metrics

    def _print_summary(self) -> None:
        """Print scalar metrics that are not necessarily plotted."""
        metrics = self.summary_metrics(last_k=200)

        print("\n=== Summary over last 200 episodes ===")
        print(
            "Cooldown (trust vs base): "
            f"{metrics['cooldown_trust']:.4f} vs {metrics['cooldown_base']:.4f}"
        )
        print(
            "Conflict (trust vs base): "
            f"{metrics['conflict_trust']:.4f} vs {metrics['conflict_base']:.4f}"
        )
        print(
            "Cluster lifetime (trust vs base): "
            f"{metrics['cluster_lifetime_trust']:.1f} vs {metrics['cluster_lifetime_base']:.1f}"
        )
        print(
            "Min return (trust vs base): "
            f"{metrics['min_return_trust']:.3f} vs {metrics['min_return_base']:.3f}"
        )
        print(
            "Cooperation index (all agents >= 0.1 reward): "
            f"{metrics['coop_index_trust']:.3f} (trust) vs {metrics['coop_index_base']:.3f} (baseline)"
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

    # Run the full experiment for multiple random seeds and report variance.
    num_seeds = 30
    all_metrics: list[dict] = []

    for i in range(num_seeds):
        seed = i + 1
        print(
            f"\n=== Running grid trust experiment (main configuration), "
            f"seed {seed}/{num_seeds} ==="
        )
        # use a per-seed suffix so that plots / CSVs are not overwritten
        exp = GridTrustExperiment(
            cfg,
            env_kwargs=env_kwargs,
            save_suffix=f"_seed{seed}",
            global_seed=seed,
        )
        exp.run()

        # collect scalar metrics for variance analysis
        m = exp.summary_metrics(last_k=200)
        m["seed"] = seed
        all_metrics.append(m)

        # After training for this seed, record GIFs (optional visualisation)
        try:
            base_dir = os.path.join(os.path.dirname(__file__), "..")

            trust_gif = os.path.join(base_dir, f"grid_trust_episode_seed{seed}.gif")
            base_gif = os.path.join(base_dir, f"grid_baseline_episode_seed{seed}.gif")
            comp_gif = os.path.join(
                base_dir, f"grid_trust_vs_baseline_seed{seed}.gif"
            )

            # record full episodes up to env.max_steps for each run
            exp.record_trust_episode_gif(out_path=trust_gif)
            exp.record_baseline_episode_gif(out_path=base_gif)
            exp.record_comparison_gif(out_path=comp_gif)

            print(f"\nSaved GIF of a trust-based episode to: {trust_gif}")
            print(f"Saved GIF of a baseline episode to:      {base_gif}")
            print(
                f"Saved comparison GIF (left=trust, right=baseline) to: {comp_gif}"
            )
        except RuntimeError as e:
            # imageio is optional, so do not crash if it is missing
            print(f"\n[Warning] Could not record GIFs for seed {seed}: {e}")

    # After all seeds are finished, compute mean / std across seeds as a
    # simple measure of variance for the main scalar metrics.
    if all_metrics:
        import csv

        print("\n=== Aggregated metrics over seeds (mean ± std over last 200 episodes) ===")
        keys = [k for k in all_metrics[0].keys() if k != "seed"]
        agg_rows = []

        for key in keys:
            vals = np.asarray([m[key] for m in all_metrics], dtype=np.float32)
            mean = float(vals.mean())
            std = float(vals.std())
            print(f"{key}: {mean:.4f} ± {std:.4f}  (n={num_seeds})")
            agg_rows.append((key, mean, std))

        # also save aggregated metrics to a small CSV for later use in the paper
        base_dir = os.path.join(os.path.dirname(__file__), "..")
        summary_csv = os.path.join(base_dir, "grid_seed_summary.csv")
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "std", "num_seeds"])
            for key, mean, std in agg_rows:
                writer.writerow([key, mean, std, num_seeds])
        print(f"\nSaved aggregated seed summary to: {summary_csv}")

    # After all seeds are finished, automatically generate aggregated plots
    # (smoothed mean curves with SEM, and difference plots)
    print("\n=== Generating aggregated plots (mean ± SEM over 30 seeds) ===")
    try:
        from .grid_trust_aggregate import main as aggregate_main
        aggregate_main()
        print("Successfully generated aggregated plots.")
    except Exception as e:
        print(f"\n[Warning] Could not generate aggregated plots: {e}")
        print("You can run 'python -m hunger_game.experiments.grid_trust_aggregate' separately to generate them.")

