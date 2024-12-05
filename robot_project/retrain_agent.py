from train_functions import retrain_agent

from greedy import list_saved_agents, select_agent

if __name__ == "__main__":
    train_config = {
        "episodes": 10000,
        "max_steps": 1000,
        "render_freq": 10,
        "render_mode": None,
        "render_delay": 0.1,
    }
    
    selected_agent = select_agent()
    if selected_agent:
        retrain_agent(selected_agent["agent"], selected_agent["env"], train_config)
    else:
        print("No agent selected. Exiting.")
