import os
import sys
import numpy as np
import pygame
from collections import defaultdict
from .visualization import Visualizer
from .metrics import Metrics

class TowerEnvironment:
    def __init__(self, config):
        """
        Initialize tower environment inspired by 'The Platform' movie
        Args:
            config: Configuration parameter dictionary
        """
        self.num_agents = 4
        self.tower_height = 4
        self.initial_food = 4

        self.hunger_increase_rate = config['environment']['hunger_increase_rate']
        self.max_hunger = config['environment']['max_hunger']
        self.min_hunger = config['environment']['min_hunger']
        
        self.rewards = {
            'survive_step': 0.1,
            'death': -1.0,
        }

        self.metrics = Metrics()

        self.visualizer = Visualizer({
            'visualization': {
                'window_width': 800,
                'window_height': 600,
                'fps': 30
            }
        })
        
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.agent_hunger = [0.0] * self.num_agents
        self.platform_position = self.tower_height
        self.remaining_food = self.initial_food

        self.agent_positions = list(range(1, self.num_agents + 1))
        np.random.shuffle(self.agent_positions)

        self.food_consumption_history = []
        return self._get_state()
    
    def step(self, actions, communication=False):
        """
        Execute one time step within the environment
        Args:
            actions: List of agent actions (0: no eat, 1: eat 1 portion, 2: eat 2 portions)
            communication: Whether agents can communicate
        """
        self.last_actions = actions.copy()
        
        rewards = []
        food_before = self.remaining_food
        
        for i in range(self.num_agents):
            self.agent_hunger[i] = min(
                self.max_hunger,
                self.agent_hunger[i] + self.hunger_increase_rate
            )
        for i, agent_floor in enumerate(self.agent_positions):
            reward = 0
            if agent_floor == self.platform_position:
                action = actions[i]
                if action > 0 and self.remaining_food >= action:
                    self.remaining_food -= action
                    self.agent_hunger[i] = max(
                        self.min_hunger,
                        self.agent_hunger[i] - 0.5 * action
                    )
                    reward += action

            if self.agent_hunger[i] >= self.max_hunger:
                reward += self.rewards['death']
            else:
                reward += self.rewards['survive_step']
            
            rewards.append(reward)
        
        self.food_consumption_history.append({
            'floor': self.platform_position,
            'food_before': food_before,
            'food_after': self.remaining_food,
            'consumers': [i for i, floor in enumerate(self.agent_positions)
                         if floor == self.platform_position and actions[i] > 0]
        })
        
        self.platform_position -= 1
        done = self.platform_position < 1
        
        if done:
            np.random.shuffle(self.agent_positions)
            self.platform_position = self.tower_height
            self.remaining_food = self.initial_food
            self.food_consumption_history = []
        
        self.last_rewards = rewards.copy()

        self.metrics.update(self)
        
        return self._get_state(communication), rewards, done, {
            'hunger': self.agent_hunger,
            'food': self.remaining_food,
            'consumption_history': self.food_consumption_history
        }
    
    def _get_state(self, communication=False):
        """Get current environment state"""
        current_agent_idx = self.agent_positions.index(self.platform_position)

        base_state = {
            'floor_id': self.platform_position,
            'remaining_food': self.remaining_food,
            'hunger_level': self.agent_hunger[current_agent_idx],
            'all_hunger_levels': self.agent_hunger.copy(),
            'all_positions': self.agent_positions.copy(),
            'consumption_per_floor': defaultdict(int)
        }
        
        if hasattr(self, 'last_actions'):
            for i, (floor, action) in enumerate(zip(self.agent_positions, self.last_actions)):
                base_state['consumption_per_floor'][floor] += action
        if communication:
            base_state.update({
                'food_history': self.food_consumption_history
            })
        
        return base_state
    
    def render(self):
        """Render the environment"""
        return self.visualizer.update(self)