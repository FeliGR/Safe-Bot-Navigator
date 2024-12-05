from train_functions import create_scenario

env_config = {
        'size': 10,
        'obstacle_prob': 0.2,
        'trap_prob': 0.2,
        'trap_danger': 0.3,
        'rewards': {
            'target': 1,
            'collision': 0,
            'step': 0,
            'trap': 0
        }
    }

train_config = {
    'episodes': 10000,
    'max_steps': 1000,
    'render_freq': 1000,
    'render_mode': None,
    'render_delay': 0.1,
}

# agent_config = {
#     'agent_module': 'agents.basic_qlearning',  # Module path to the agent
#     'agent_class': 'BasicQLearningAgent',       # Class name of the agent
#     'learning_rate': 0.1,
#     'gamma': 0.99,
#     'epsilon': 1.0,
#     'epsilon_min': 0.01        
# }

agent_config = {
    'agent_module': 'agents.q_learning_marie_2005_planifier',
    'agent_class': 'QLearningAgentMarie2005',
    'learning_rate': 0.1,
    'gamma': 0.99,
    'epsilon': 0,
    'epsilon_min': 0,
    'epsilon_decay': 0,
    'planified_episodes': 100
}

create_scenario(env_config, agent_config, train_config)
