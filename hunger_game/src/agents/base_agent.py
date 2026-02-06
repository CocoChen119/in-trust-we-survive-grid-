class BaseAgent:
    def __init__(self, state_space, action_space, communication=False):
        """
        Initialize base agent
        Args:
            state_space: State space definition
            action_space: Action space definition
            communication: Whether the agent has communication
        """
        self.state_space = state_space
        self.action_space = action_space
        self.communication = communication
    
    def select_action(self, state):
        """
        Select action based on current state
        Args:
            state: Current environment state
        Returns:
            Selected action
        """
        raise NotImplementedError
    
    def learn(self, state, action, reward, next_state):
        """
        Learn from experience
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        raise NotImplementedError
    
    def get_state_key(self, state):
        """Base method for state key generation"""
        raise NotImplementedError
