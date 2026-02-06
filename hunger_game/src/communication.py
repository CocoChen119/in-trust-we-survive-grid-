class CommunicationSystem:
    def __init__(self, num_agents):
        """
        Initialize communication system
        Args:
            num_agents: Number of agents in the system
        """
        self.num_agents = num_agents
        self.messages = []
        
    def broadcast(self, agent_id, message):
        """
        Broadcast message to all agents
        Args:
            agent_id: ID of the sending agent
            message: Message content (food status)
        """
        self.messages.append({
            'sender': agent_id,
            'content': message
        })
        
    def receive(self, agent_id):
        """
        Receive messages from other agents
        Args:
            agent_id: ID of the receiving agent
        Returns:
            List of messages from other agents
        """
        return [msg for msg in self.messages if msg['sender'] != agent_id]