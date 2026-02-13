import json
import os

os.makedirs('config', exist_ok=True)

# 配置内容
config = {
    "environment": {
        "num_floors": 4,
        "initial_food": 4,
        "hunger_increase_rate": 0.5,
        "max_hunger": 2.0,
        "min_hunger": 0.0
    },
    "training": {
        "episodes": 8000,
        "eval_interval": 100,
        "q_learning": {
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "epsilon": 0.1
        },
        "monte_carlo": {
            "epsilon": 0.1
        },
        "random_seed": 42
    },
    "rewards": {
        "survive_step": 0.1,
        "death": -1.0
    },
    "visualization": {
        "window_width": 400,
        "window_height": 600,
        "fps": 30
    }
}

with open('config/config.json', 'w') as f:
    json.dump(config, f, indent=4)

print("Configuration file created successfully!")
