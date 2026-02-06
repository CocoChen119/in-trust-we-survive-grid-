import numpy as np

from collections import defaultdict
from .base_agent import BaseAgent

class MonteCarloAgent(BaseAgent):
    def __init__(self, state_space, action_space, epsilon=0.1, communication=False):
        """Initialize Monte Carlo agent with communication option"""
        super().__init__(state_space, action_space)
        self.q_table = defaultdict(lambda: np.zeros(len(action_space)))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.epsilon = epsilon
        self.episode_memory = []
        self.communication = communication
        print(f"Initialized Monte Carlo agent with epsilon={epsilon}")
    
    def get_state_key(self, state):
        """Convert state to hashable key based on communication mode"""
        if self.communication:
            return (
                state['floor_id'],
                state['remaining_food'],
                tuple(state['all_hunger_levels']),
                tuple(state['all_positions']),
                tuple(sorted(state.get('consumption_per_floor', {}).items()))
            )
        else:
            return (
                state['floor_id'],
                state['remaining_food'],
                state['hunger_level']
            )
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.action_space))
        
        return np.argmax(self.q_table[state_key])
    
    def store_transition(self, state, action, reward):
        """
        Store state-action-reward transition
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
        """
        self.episode_memory.append((state, action, reward))
    
    def learn(self, state, action, reward, next_state):
        """Store experience for Monte Carlo update"""
        self.episode_memory.append((self.get_state_key(state), action, reward))
        
        if next_state is None or next_state.get('floor_id', 0) <= 1:
            self._update_policy()

    def _update_policy(self):
        """Update policy using Monte Carlo method"""
        if not self.episode_memory:
            return
            
        G = 0
        for t in reversed(range(len(self.episode_memory))):
            state_key, action, reward = self.episode_memory[t]
            G = reward + G
            
            self.returns[state_key][action].append(G)
            self.q_table[state_key][action] = np.mean(self.returns[state_key][action])
        
        self.episode_memory = []
