import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict
from src.environment import TowerEnvironment
from src.agents.qianpu import ImitationEvoAgent
from copy import deepcopy

class UltraLowTrustIEGS(ImitationEvoAgent):
    def __init__(self, state_space, action_space, epsilon=0.1, communication=False):
        super().__init__(state_space, action_space, epsilon, communication)
        self.trust = defaultdict(lambda: -1.0)

class LowTrustIEGS(ImitationEvoAgent):
    def __init__(self, state_space, action_space, epsilon=0.1, communication=False):
        super().__init__(state_space, action_space, epsilon, communication)
        self.trust = defaultdict(lambda: -0.5)

class NeutralTrustIEGS(ImitationEvoAgent):
    def __init__(self, state_space, action_space, epsilon=0.1, communication=False):
        super().__init__(state_space, action_space, epsilon, communication)
        self.trust = defaultdict(lambda: 0.0)
class ShortGreedyIEGS(ImitationEvoAgent):
    def __init__(self, state_space, action_space, epsilon=0.1, communication=False):
        super().__init__(state_space, action_space, epsilon, communication)
        self.forced_greedy_episodes = 200
        self.current_episode = 0
    
    def select_action(self, state):
        if self.current_episode < self.forced_greedy_episodes:
            return min(2, state['remaining_food'])
        return super().select_action(state)
    
    def learn(self, state, action, reward, next_state):
        super().learn(state, action, reward, next_state)
        if next_state['floor_id'] == self.state_space['floor_id']:
            self.current_episode += 1

class MediumGreedyIEGS(ImitationEvoAgent):
    def __init__(self, state_space, action_space, epsilon=0.1, communication=False):
        super().__init__(state_space, action_space, epsilon, communication)
        self.forced_greedy_episodes = 600
        self.current_episode = 0
    
    def select_action(self, state):
        if self.current_episode < self.forced_greedy_episodes:
            return min(2, state['remaining_food'])
        return super().select_action(state)
    
    def learn(self, state, action, reward, next_state):
        super().learn(state, action, reward, next_state)
        if next_state['floor_id'] == self.state_space['floor_id']:
            self.current_episode += 1

class LongGreedyIEGS(ImitationEvoAgent):
    def __init__(self, state_space, action_space, epsilon=0.1, communication=False):
        super().__init__(state_space, action_space, epsilon, communication)
        self.forced_greedy_episodes = 800
        self.current_episode = 0
    
    def select_action(self, state):
        if self.current_episode < self.forced_greedy_episodes:
            return min(2, state['remaining_food'])
        return super().select_action(state)
    
    def learn(self, state, action, reward, next_state):
        super().learn(state, action, reward, next_state)
        if next_state['floor_id'] == self.state_space['floor_id']:
            self.current_episode += 1

class HighFloorGreedyIEGS(ImitationEvoAgent):
    def __init__(self, state_space, action_space, forced_episodes=800, epsilon=0.1, communication=False):
        super().__init__(state_space, action_space, epsilon, communication)
        self.forced_greedy_episodes = forced_episodes
        self.current_episode = 0
    
    def select_action(self, state):
        if self.current_episode < self.forced_greedy_episodes:
            if state['floor_id'] > self.state_space['floor_id'] // 2:
                return min(2, state['remaining_food'])
        return super().select_action(state)
    
    def learn(self, state, action, reward, next_state):
        super().learn(state, action, reward, next_state)
        if next_state['floor_id'] == self.state_space['floor_id']:
            self.current_episode += 1

def run_variant_experiments(config):
    """Run IEGS variant experiments."""
    variants = {
        'Ultra Low Trust IEGS': UltraLowTrustIEGS,
        'Low Trust IEGS': LowTrustIEGS,
        'Neutral Trust IEGS': NeutralTrustIEGS,
        
        'Short Greedy IEGS': ShortGreedyIEGS,
        'Medium Greedy IEGS': MediumGreedyIEGS,
        'Long Greedy IEGS': LongGreedyIEGS,
        
        'High Floor Greedy IEGS': HighFloorGreedyIEGS
    }
    
    results = {}
    
    for variant_name, variant_class in variants.items():
        print(f"\nStarting {variant_name} experiment...")
        env = TowerEnvironment(config)
        
        state_space = {
            'floor_id': env.tower_height,
            'remaining_food': env.initial_food,
            'hunger_level': env.max_hunger
        }
        action_space = [0, 1, 2]
        
        agents = [
            variant_class(state_space, action_space)
            for _ in range(env.num_agents)
        ]
        
        # 训练循环
        for episode in range(config['training']['episodes']):
            state = env.reset()
            done = False
            
            while not done:
                actions = [agent.select_action(state) for agent in agents]
                next_state, rewards, done, info = env.step(actions)
                
                for agent, action, reward in zip(agents, actions, rewards):
                    agent.learn(state, action, reward, next_state)
                
                state = next_state
            
            if episode % 100 == 0:
                print(f"Episode {episode}/{config['training']['episodes']}")
                metrics = env.metrics.metrics
                print(f"Survival Rate: {metrics['survival_rate'][-1]:.3f}")
                print(f"Success Rate: {metrics['success_rate'][-1]:.3f}")
        
        results[variant_name] = env.metrics

        env.metrics.plot_early_training(
            title=f"{variant_name} Early Training Phase",
            save_path=f"{variant_name.lower().replace(' ', '_')}_early.png",
            algo_name=variant_name,
            num_episodes=1000
        )
        
        env.metrics.plot_metrics(
            title=f"{variant_name} Results",
            save_path=f"{variant_name.lower().replace(' ', '_')}_results.png",
            algo_name=variant_name
        )
    
    return results

if __name__ == "__main__":
    config = {
        "environment": {
            "num_floors": 4,
            "initial_food": 4,
            "hunger_increase_rate": 0.5,
            "max_hunger": 2.0,
            "min_hunger": 0.0
        },
        "training": {"episodes": 5000, "eval_interval": 100}
    }
    
    results = run_variant_experiments(config)
