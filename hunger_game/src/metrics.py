import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class Metrics:
    def __init__(self):
        """Initialize metrics tracking"""
        self.reset()
        self.SMOOTH_WINDOW = 200
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'episode_rewards': [],
            'survival_rate': [],
            'success_rate': [],
            'hunger_levels': [],
            'consumption_efficiency': [],
            'fairness_index': [],
            'cooperation_level': [],
            'system_stability': [],
            'resource_utilization': [],
        }
    
    def update(self, env):
        """Update metrics based on current environment state."""
        alive_agents = sum(1 for hunger in env.agent_hunger if hunger < env.max_hunger)
        self.metrics['survival_rate'].append(alive_agents / env.num_agents)
        
        if hasattr(env, 'last_actions'):
            moderate_agents = sum(1 for action in env.last_actions if action <= 1)
            self.metrics['success_rate'].append(moderate_agents / env.num_agents)
        
        if hasattr(env, 'last_rewards'):
            self.metrics['episode_rewards'].append(np.mean(env.last_rewards))
        
        self.metrics['hunger_levels'].append(np.mean(env.agent_hunger))
        
        if env.food_consumption_history:
            last_consumption = env.food_consumption_history[-1]
            efficiency = (last_consumption['food_before'] - last_consumption['food_after']) / env.initial_food
            self.metrics['consumption_efficiency'].append(efficiency)
        
        hunger_values = np.array(env.agent_hunger)
        fairness = 1 - (np.std(hunger_values) / env.max_hunger)
        self.metrics['fairness_index'].append(fairness)
        
        if hasattr(env, 'last_actions'):
            cooperative_actions = sum(1 for action in env.last_actions if action == 1)
            self.metrics['cooperation_level'].append(cooperative_actions / env.num_agents)
        
        stability = 1 - np.std(env.agent_hunger) / env.max_hunger
        self.metrics['system_stability'].append(stability)
        
        utilization = 1 - (env.remaining_food / env.initial_food)
        self.metrics['resource_utilization'].append(utilization)
    
    def smooth_data(self, data):
        """Smooth time-series data with a moving average."""
        if len(data) < self.SMOOTH_WINDOW:
            return data
        return np.convolve(data, 
                          np.ones(self.SMOOTH_WINDOW)/self.SMOOTH_WINDOW, 
                          mode='valid')
    
    def plot_metrics(self, title="Training Metrics", save_path=None, algo_name=""):
        """Plot all metrics"""
        metrics_config = {
            'survival_rate': {'title': 'Survival Rate', 'color': 'blue'},
            'success_rate': {'title': 'Success Rate', 'color': 'green'},
            'hunger_levels': {'title': 'Hunger Level', 'color': 'red'},
            'consumption_efficiency': {'title': 'Consumption Efficiency', 'color': 'purple'},
            'fairness_index': {'title': 'Fairness Index', 'color': 'orange'},
            'cooperation_level': {'title': 'Cooperation Level', 'color': 'brown'},
            'system_stability': {'title': 'System Stability', 'color': 'cyan'},
            'resource_utilization': {'title': 'Resource Utilization', 'color': 'pink'},
            'episode_rewards': {'title': 'Average Rewards', 'color': 'gray'}
        }
        
        num_metrics = len(metrics_config)
        num_rows = (num_metrics + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5*num_rows))
        fig.suptitle(f'{algo_name} Training Results', fontsize=16)
        plt.subplots_adjust(top=0.95)
        
        for idx, (metric_name, config) in enumerate(metrics_config.items()):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col] if num_rows > 1 else axes[col]
            
            if self.metrics[metric_name]:
                data = np.array(self.metrics[metric_name])
                smoothed_data = self.smooth_data(data)
                episodes = np.arange(len(smoothed_data))
                
                ax.plot(episodes, smoothed_data, 
                       color=config['color'], 
                       linewidth=2, 
                       label=config['title'])
                ax.fill_between(episodes,
                              smoothed_data - np.std(data),
                              smoothed_data + np.std(data),
                              color=config['color'],
                              alpha=0.1)
                
                ax.set_title(config['title'])
                ax.set_xlabel('Episodes')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_final_metrics(self):
        """Get average of last 100 episodes for each metric."""
        return {
            metric: np.mean(values[-100:])
            for metric, values in self.metrics.items()
        }

    def compare_metrics(self, comm_metrics, no_comm_metrics, save_path=None):
        """Compare metrics with and without communication."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Metrics Comparison: Communication vs No Communication')
        
        metrics_config = {
            'success_rate': {'title': 'Survival Rate', 'color': ['green', 'lightgreen']},
            'episode_rewards': {'title': 'Episode Rewards', 'color': ['red', 'pink']},
            'cooperation': {'title': 'Cooperation Index', 'color': ['purple', 'plum']},
            'fairness': {'title': 'Fairness Index', 'color': ['orange', 'peachpuff']}
        }
        
        for idx, (metric_name, config) in enumerate(metrics_config.items()):
            row = idx // 3
            col = idx % 3
            if metric_name in comm_metrics.metrics and metric_name in no_comm_metrics.metrics:
                axes[row, col].plot(comm_metrics.metrics[metric_name], 
                                  color=config['color'][0], 
                                  label='With Comm')
                axes[row, col].plot(no_comm_metrics.metrics[metric_name], 
                                  color=config['color'][1], 
                                  label='No Comm')
                axes[row, col].set_title(config['title'])
                axes[row, col].set_xlabel('Episodes')
                axes[row, col].set_ylabel('Value')
                axes[row, col].legend()
                axes[row, col].grid(True)
        
        if len(metrics_config) < 6:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_algorithm_comparison(self, metrics_dict, title="Algorithm Comparison", save_path=None):
        """Plot comparison between different algorithms"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.95)
        
        colors = {
            'Q-Learning': '#1f77b4',
            'Monte Carlo': '#2ca02c',
            'IE-GS': '#d62728'
        }
        
        for algo_name, metrics in metrics_dict.items():
            if metrics.metrics['episode_rewards']:
                rewards = np.array(metrics.metrics['episode_rewards'])
                smoothed_rewards = self.smooth_data(rewards)
                episodes = np.arange(len(smoothed_rewards))
                axes[0, 0].plot(episodes, smoothed_rewards, 
                              label=algo_name, 
                              color=colors.get(algo_name, 'gray'),
                              linewidth=2)
        
        axes[0, 0].set_title('Average Rewards', fontsize=12)
        axes[0, 0].set_xlabel('Episodes', fontsize=10)
        axes[0, 0].set_ylabel('Reward', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=10)
        
        for algo_name, metrics in metrics_dict.items():
            if metrics.metrics['survival_rate']:
                survival_rates = np.array(metrics.metrics['survival_rate'])
                smoothed_rates = self.smooth_data(survival_rates)
                episodes = np.arange(len(smoothed_rates))
                axes[0, 1].plot(episodes, smoothed_rates, 
                              label=algo_name, 
                              color=colors.get(algo_name, 'gray'),
                              linewidth=2)
        
        axes[0, 1].set_title('Survival Rate', fontsize=12)
        axes[0, 1].set_xlabel('Episodes', fontsize=10)
        axes[0, 1].set_ylabel('Rate', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=10)
        
        for algo_name, metrics in metrics_dict.items():
            if metrics.metrics['success_rate']:
                success_rates = np.array(metrics.metrics['success_rate'])
                smoothed_rates = self.smooth_data(success_rates)
                episodes = np.arange(len(smoothed_rates))
                axes[1, 0].plot(episodes, smoothed_rates, 
                              label=algo_name, 
                              color=colors.get(algo_name, 'gray'),
                              linewidth=2)
        
        axes[1, 0].set_title('Success Rate (Moderation)', fontsize=12)
        axes[1, 0].set_xlabel('Episodes', fontsize=10)
        axes[1, 0].set_ylabel('Rate', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_smoothed_metrics(self):
        """Return smoothed version of all metrics."""
        return {
            name: self.smooth_data(np.array(values))
            for name, values in self.metrics.items()
        }

    def plot_early_training(self, title="Early Training Phase", save_path=None, algo_name="", num_episodes=1000):
        """Plot metrics for early training phase"""
        metrics_config = {
            'survival_rate': {'title': 'Early Survival Rate', 'color': 'blue'},
            'success_rate': {'title': 'Early Success Rate', 'color': 'green'},
            'episode_rewards': {'title': 'Early Average Rewards', 'color': 'red'},
            'hunger_levels': {'title': 'Early Hunger Level', 'color': 'orange'}
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{algo_name} Early Training Phase (First {num_episodes} Episodes)', fontsize=16)
        
        for idx, (metric_name, config) in enumerate(metrics_config.items()):
            row = idx // 2
            col = idx % 2
            
            if self.metrics[metric_name]:
                data = np.array(self.metrics[metric_name][:num_episodes])
                episodes = np.arange(len(data))
                
                small_window = min(50, len(data) // 10)
                if len(data) > small_window:
                    smoothed_data = np.convolve(data, 
                                              np.ones(small_window)/small_window, 
                                              mode='valid')
                    episodes = np.arange(len(smoothed_data))
                else:
                    smoothed_data = data
                
                axes[row, col].plot(episodes, smoothed_data, 
                                  color=config['color'], 
                                  linewidth=2, 
                                  label='Smoothed')
                axes[row, col].scatter(np.arange(len(data)), data, 
                                     color=config['color'], 
                                     alpha=0.1, 
                                     s=1, 
                                     label='Raw')
                
                axes[row, col].set_title(config['title'])
                axes[row, col].set_xlabel('Episodes')
                axes[row, col].set_ylabel('Value')
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].legend()
        
        plt.tight_layout()
        if save_path:
            base, ext = os.path.splitext(save_path)
            early_save_path = f"{base}_early{ext}"
            plt.savefig(early_save_path, dpi=300, bbox_inches='tight')
        plt.show()
