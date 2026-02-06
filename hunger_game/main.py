import json
import numpy as np
from src.environment import TowerEnvironment
from src.agents.q_learning import QLearningAgent
from src.agents.monte_carlo import MonteCarloAgent
from src.agents.qianpu import ImitationEvoAgent
import matplotlib.pyplot as plt
import os
import glob
import pygame
import argparse

def load_config():
    """Load configuration from JSON file"""
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        print("Configuration loaded successfully")
        return config
    except FileNotFoundError:
        print("Config file not found. Using default configuration.")
        return {
            "environment": {
                "num_floors": 4,
                "initial_food": 4,
                "hunger_increase_rate": 0.5,
                "max_hunger": 2.0,
                "min_hunger": 0.0
            },
            "training": {
                "episodes": 8000,
                "eval_interval": 100,
                "q_learning": {
                    "learning_rate": 0.1,
                    "discount_factor": 0.95,
                    "epsilon": 0.1
                },
                "monte_carlo": {
                    "epsilon": 0.1
                },
                "random_seed": 42
            },
            "rewards": {
                "survive_step": 0.1,
                "death": -1.0
            }
        }

def train(env, agent_type, config, communication=False, render=False):
    """Train agents with or without communication"""
    state_space = {
        'floor_id': env.tower_height,
        'remaining_food': env.initial_food,
        'hunger_level': env.max_hunger
    }
    action_space = [0, 1, 2]

    env.metrics.reset()

    agents = [
        agent_type(state_space, action_space, communication=communication)
        for _ in range(env.num_agents)
    ]
    
    for episode in range(config['training']['episodes']):
        state = env.reset()
        done = False
        episode_rewards = []
        while not done:
            actions = [agent.select_action(state) for agent in agents]
            next_state, rewards, done, info = env.step(actions, communication)
            if render:
                env.render()
                pygame.time.wait(500)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return agents, env.metrics
            for agent, action, reward in zip(agents, actions, rewards):
                agent.learn(state, action, reward, next_state)
                episode_rewards.append(reward)
            
            state = next_state
        if episode % config['training']['eval_interval'] == 0:
            avg_reward = np.mean(episode_rewards)
            success_rate = env.metrics.metrics['success_rate'][-1] if env.metrics.metrics['success_rate'] else 0
            print(f"\nEpisode {episode}/{config['training']['episodes']}")
            print(f"Average Reward: {avg_reward:.3f}")
            print(f"Success Rate: {success_rate:.3f}")
            for metric_name, values in env.metrics.metrics.items():
                if values:
                    print(f"{metric_name}: {values[-1]:.3f}")
    
    if render:
        pygame.quit()
    
    return agents, env.metrics

def clean_previous_results():
    """Clean up previous training results"""
    for file in glob.glob("*.png"):
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
    results_dir = "results"
    if os.path.exists(results_dir):
        for file in glob.glob(f"{results_dir}/*.png"):
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")

def main():
    clean_previous_results()
    
    config = load_config()
    np.random.seed(config['training']['random_seed'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Enable visualization')
    args = parser.parse_args()
    
    print("\nStarting Q-Learning experiment...")
    env_q = TowerEnvironment(config)
    agents_q, metrics_q = train(env_q, QLearningAgent, config, render=args.render)
    
    print("\nStarting Monte Carlo experiment...")
    env_mc = TowerEnvironment(config)
    agents_mc, metrics_mc = train(env_mc, MonteCarloAgent, config, render=args.render)
    
    print("\nStarting IE-GS experiment...")
    env_ie = TowerEnvironment(config)
    agents_ie, metrics_ie = train(env_ie, ImitationEvoAgent, config, render=args.render)
    
    metrics_ie.plot_early_training(
        title="IE-GS Early Training Phase",
        save_path="ie-gs_early_results.png",
        algo_name="IE-GS",
        num_episodes=1000,
    )
    
    metrics_dict = {
        'Q-Learning': metrics_q,
        'Monte Carlo': metrics_mc,
        'IE-GS': metrics_ie
    }
    
    for name, metrics in metrics_dict.items():
        metrics.plot_metrics(
            title=f"{name} Results",
            save_path=f"{name.lower()}_results.png",
            algo_name=name
        )
    
    metrics_q.plot_algorithm_comparison(
        metrics_dict,
        title="Algorithm Comparison",
        save_path="algorithm_comparison.png"
    )

if __name__ == "__main__":
    main()