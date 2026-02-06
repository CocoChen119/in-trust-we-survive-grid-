import numpy as np
from collections import defaultdict, deque
from .base_agent import BaseAgent

class ImitationEvoAgent(BaseAgent):
    def __init__(self, state_space, action_space, epsilon=0.1, communication=False):
        """Initialize Imitation Evolutionary Game Strategy agent"""
        super().__init__(state_space, action_space)
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.communication = communication

        self.trust = defaultdict(float)
        self.trust_history = defaultdict(lambda: deque(maxlen=100))

        self.short_memory = deque(maxlen=10)
        self.long_memory = deque(maxlen=1000)
        self.strategy = np.random.choice(action_space)

        self.max_hunger = state_space['hunger_level']
        self.max_food = state_space['remaining_food']

        self.performance_history = deque(maxlen=100)
        print(f"Initialized IE-GS agent with epsilon={epsilon}")

    def select_action(self, state):
        """Select action based on trust and exploration"""
        if self._should_explore():
            return np.random.choice(self.action_space)
        
        current_floor = state['floor_id']
        next_floor = current_floor - 1 if current_floor > 1 else None
        hunger_level = state['hunger_level']
        remaining_food = state['remaining_food']
        
        if next_floor and next_floor in self.trust:
            trust_value = self.trust[next_floor]

            if trust_value > 0.5:
                if hunger_level > self.max_hunger * 0.6:
                    return min(1, remaining_food)
                return 0

            elif trust_value < -0.3:
                if hunger_level > self.max_hunger * 0.8:
                    return min(2, remaining_food)
                return min(1, remaining_food)

        return min(self._calculate_optimal_action(state), remaining_food)

    def learn(self, state, action, reward, next_state):
        """Update trust values and strategy based on observations"""
        self.short_memory.append((state, action, reward))
        self.long_memory.append((state, action, reward))

        if next_state:
            self._update_trust(state, next_state, action, reward)

        recent_rewards = list(self.short_memory)[-5:]
        recent_performance = np.mean([r for _, _, r in recent_rewards]) if recent_rewards else 0
        
        long_term_rewards = list(self.long_memory)
        long_term_performance = np.mean([r for _, _, r in long_term_rewards]) if long_term_rewards else 0

        if recent_performance < long_term_performance:
            self.epsilon = min(0.9, self.epsilon * 1.1)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)
        self.performance_history.append(reward)

    def _update_trust(self, state, next_state, action, reward):
        """Update trust values based on behavior observation"""
        current_floor = state['floor_id']
        next_floor = current_floor - 1
        
        if next_floor > 0:
            trust_update = 0

            food_ratio = next_state['remaining_food'] / state['remaining_food'] if state['remaining_food'] > 0 else 0
            if food_ratio > 0.5:
                trust_update += 0.2
            elif food_ratio < 0.2:
                trust_update -= 0.2

            if self.communication:
                avg_hunger = np.mean(next_state['all_hunger_levels'])
                if avg_hunger < self.max_hunger * 0.6:
                    trust_update += 0.1
                elif avg_hunger > self.max_hunger * 0.8:
                    trust_update -= 0.1
            
            self.trust[next_floor] = np.clip(self.trust[next_floor] + trust_update, -1.0, 1.0)
            self.trust_history[next_floor].append(trust_update)

    def _calculate_optimal_action(self, state):
        """Calculate optimal action based on current state"""
        hunger_level = state['hunger_level']
        remaining_food = state['remaining_food']
        
        if hunger_level > self.max_hunger * 0.8:
            return min(2, remaining_food)
        elif hunger_level > self.max_hunger * 0.5:
            return min(1, remaining_food)
        else:
            return 0

    def _should_explore(self):
        """Determine if agent should explore based on performance"""
        if len(self.performance_history) < 10:
            return np.random.random() < self.epsilon
        
        recent_performance = np.mean(list(self.performance_history))
        if recent_performance < 0:
            return np.random.random() < self.epsilon
        return np.random.random() < self.epsilon_min