import torch
import torch.nn as nn
import numpy as np
from src.agents.q_learning import QLearningAgent
from src.agents.monte_carlo import MonteCarloAgent
import pygame

class Agent(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Initialize agent
        Args:
            state_size: Size of state space
            action_size: Size of action space
        """
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, state):
        """
        Input state, output action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        return torch.argmax(self.network(state)).item()
        
def train(env, agent_type, config, communication=False):
    """Train agents
    Args:
        env: Environment instance
        agent_type: 'q_learning' or 'monte_carlo'
        config: Configuration dictionary
        communication: Whether to enable communication between agents
    """
    print(f"\n=== Starting Training: {agent_type} {'with' if communication else 'without'} communication ===")
    
    # Create agents
    agents = []
    for i in range(config['agent']['num_agents']):
        if agent_type == 'q_learning':
            agent = QLearningAgent(
                state_space=None,
                action_space=range(3),
                learning_rate=config['training']['q_learning']['learning_rate'],
                discount_factor=config['training']['q_learning']['discount_factor'],
                epsilon=config['training']['q_learning']['epsilon']
            )
        else:
            agent = MonteCarloAgent(
                state_space=None,
                action_space=range(3),
                epsilon=config['training']['monte_carlo']['epsilon']
            )
        agents.append(agent)
    
    # Training loop
    for episode in range(config['training']['episodes']):
        state = env.reset()
        done = False
        
        while not done:
            actions = []
            for i, agent in enumerate(agents):
                agent_state = state.copy()
                if not communication:
                    agent_state['my_hunger'] = env.agent_hunger[i]
                actions.append(agent.select_action(agent_state))
            
            next_state, rewards, done, info = env.step(actions, communication)
            
            # Update agents
            for i, (agent, action, reward) in enumerate(zip(agents, actions, rewards)):
                agent_next_state = next_state.copy()
                if not communication:
                    agent_next_state['my_hunger'] = env.agent_hunger[i]
                
                if agent_type == 'q_learning':
                    agent.learn(state, action, reward, agent_next_state)
                else:
                    agent.store_transition(state, action, reward)
            
            state = next_state
        
        # Monte Carlo learns at end of episode
        if agent_type == 'monte_carlo':
            for agent in agents:
                agent.learn()
        
        # Print progress
        if episode % config['metrics']['log_interval'] == 0:
            print(f"\nEpisode {episode}/{config['training']['episodes']}")
            print(f"Average Reward: {np.mean(env.last_rewards):.3f}")
            print(f"Success Rate: {env.metrics.get_latest('success_rate'):.3f}")
    
    return agents, env.metrics
        