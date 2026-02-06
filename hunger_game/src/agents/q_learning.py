import numpy as np
from .base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, state_space, action_space, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1, communication=False):
        """Initialize Q-Learning agent with communication option"""
        super().__init__(state_space, action_space)
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.communication = communication
    
    def get_state_key(self, state):
        """Convert state to hashable key based on communication mode"""
        if self.communication:
            return (
                state['floor_id'],
                state['remaining_food'],
                tuple(state['all_hunger_levels']),
                tuple(state['all_positions']),
                tuple(state.get('consumption_per_floor', {}).values())
            )
        else:
            return (state['floor_id'], state['remaining_food'], state['hunger_level'])
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy
        Args:
            state: Current state
        Returns:
            Selected action index
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.action_space))
        
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_space))
        
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state):
        """
        Update Q-values using Q-learning algorithm
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_space))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.action_space))
        
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        
        # Q-learning update formula
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q

    def train(self, num_episodes):
        rewards_history: list[float] = []
        losses: list[float] = []
        avg_q_values: list[float] = []
        # Placeholder: training loop should be implemented or this method removed
        return {"rewards": rewards_history, "losses": losses, "avg_q_values": avg_q_values}
