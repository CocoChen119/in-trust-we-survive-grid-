import pygame
import os
import numpy as np

class Visualizer:
    def __init__(self, config):
        """Initialize visualizer"""
        pygame.init()
        self.config = config
        self.width = 400
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("The Platform Simulation")
        
        self.images = self._load_images()

        self.COLORS = {
            'background': (255, 255, 255),
            'floor': (139, 69, 19),
            'platform': (200, 200, 200),
        }
        
    def _load_images(self):
        """Load all image assets."""
        images = {}
        
        try:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for agent_id in range(1, 5):
                for action in range(3):
                    key = f'agent{agent_id}-{action}'
                    path = os.path.join(current_dir, 'assets', 'agents', f'{key}.JPG')
                    images[key] = pygame.image.load(path)

            food_path = os.path.join(current_dir, 'assets', 'food', 'food.JPG')
            images['food'] = pygame.image.load(food_path)
            
            print("Successfully loaded all images")
            
        except pygame.error as e:
            print(f"Error loading images: {e}")
            print("Please check if all images are in the correct path:")
            print(f"Looking in: {current_dir}/assets/")
            print("Current working directory:", os.getcwd())
            raise
        
        return images
    
    def update(self, env):
        """Update visualization."""
        self.screen.fill(self.COLORS['background'])

        floor_height = self.height / env.tower_height

        for floor in range(env.tower_height, 0, -1):
            display_y = self.height - floor * floor_height
            pygame.draw.line(self.screen, self.COLORS['floor'], 
                           (0, display_y), (self.width * 0.7, display_y), 10)

        platform_width = self.width * 0.25
        platform_x = self.width * 0.7
        platform_y = self.height - (env.platform_position * floor_height)
        platform_rect = pygame.Rect(
            platform_x,
            platform_y + floor_height * 0.25,
            platform_width,
            floor_height * 0.5
        )
        pygame.draw.rect(self.screen, self.COLORS['platform'], platform_rect)

        food_img = self.images['food']
        food_size = min(floor_height * 0.3, platform_width * 0.2)
        scaled_food = pygame.transform.scale(food_img, (food_size, food_size))

        for i in range(int(env.remaining_food)):
            food_x = platform_x + (i + 0.5) * (platform_width / env.initial_food)
            food_y = platform_y + floor_height * 0.35
            self.screen.blit(scaled_food, (food_x, food_y))
        agent_x = self.width * 0.2
        for i, floor in enumerate(env.agent_positions):
            display_y = self.height - (floor * floor_height)
            agent_y = display_y + 10
            action = env.last_actions[i] if hasattr(env, 'last_actions') else 0
            agent_img = self.images[f'agent{i+1}-{action}']
            
            agent_size = floor_height * 0.8
            scaled_agent = pygame.transform.scale(agent_img, (agent_size, agent_size))
            self.screen.blit(scaled_agent, (agent_x, agent_y))
        pygame.display.flip()