import json
import os
import time
import numpy as np
from train_functions import train_and_compare
from environment.environment_risk import GridEnvironment

# Configuración del entorno
env_config = {
    "size": 10,
    "obstacle_prob": 0.2,
    "trap_prob": 0.2,
    "trap_danger": 0.3,
    "rewards": {"target": 1, "collision": 0, "step": -0.001, "trap": -0.2},
}

# Configuración del entrenamiento
train_config = {
    "episodes": 10000,
    "max_steps": 1000,
    "render_freq": 1000,
    "render_mode": None,
    "render_delay": 0.1,
}

# Configuración de los agentes
agent_config_no_risk = {
    "agent_module": "agents.basic_qlearning_risk",
    "agent_class": "BasicQLearningAgent",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0.8,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.000099,  # Ajusta según corresponda
    "xi": 0.0,  # Sin sensibilidad al riesgo
    "state_size": env_config["size"] ** 2,
    "n_actions": 4,  # Asumiendo 4 acciones (arriba, abajo, izquierda, derecha)
}

agent_config_with_risk = {
    "agent_module": "agents.basic_qlearning",
    "agent_class": "BasicQLearningAgent",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.000099,  # Ajusta según corresponda
    "xi": 5,  # Con sensibilidad al riesgo
    "state_size": env_config["size"] ** 2,
    "n_actions": 4,  # Asumiendo 4 acciones
}


# Generar una cuadrícula única
def generate_unique_grid(config):
    """Genera una cuadrícula única basada en la configuración proporcionada."""
    size = config["size"]
    obstacle_prob = config["obstacle_prob"]
    trap_prob = config["trap_prob"]

    grid = np.zeros((size, size), dtype=int)

    # Colocar obstáculos
    for i in range(size):
        for j in range(size):
            if (i != 0 or j != 0) and np.random.random() < obstacle_prob:
                # Verificar no tener obstáculos adyacentes
                if not _has_adjacent_obstacle_static(grid, (i, j), size):
                    grid[i, j] = GridEnvironment.OBSTACLE

    # Seleccionar posición de TARGET
    empty_cells = [
        (i, j)
        for i in range(size)
        for j in range(size)
        if grid[i, j] == GridEnvironment.EMPTY and (i != 0 or j != 0)
    ]

    if empty_cells:
        target_i, target_j = empty_cells[np.random.randint(len(empty_cells))]
        grid[target_i, target_j] = GridEnvironment.TARGET
    else:
        raise ValueError("No hay celdas vacías disponibles para colocar el TARGET.")

    # Colocar trampas
    empty_cells = [
        (i, j)
        for i in range(size)
        for j in range(size)
        if grid[i, j] == GridEnvironment.EMPTY and (i != 0 or j != 0)
    ]

    for i, j in empty_cells:
        if np.random.random() < trap_prob:
            if not _has_nearby_trap_static(grid, (i, j), size):
                grid[i, j] = GridEnvironment.TRAP

    return grid


def _has_adjacent_obstacle_static(grid, pos, size):
    """Check si una posición tiene obstáculos adyacentes en una cuadrícula estática."""
    i, j = pos
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if (
                0 <= ni < size
                and 0 <= nj < size
                and grid[ni, nj] == GridEnvironment.OBSTACLE
            ):
                return True
    return False


def _has_nearby_trap_static(grid, pos, size, range_size=2):
    """Check si una posición tiene trampas cercanas en una cuadrícula estática."""
    i, j = pos
    for di in range(-range_size + 1, range_size):
        for dj in range(-range_size + 1, range_size):
            ni, nj = i + di, j + dj
            if (
                0 <= ni < size
                and 0 <= nj < size
                and grid[ni, nj] == GridEnvironment.TRAP
            ):
                return True
    return False


# Generar la cuadrícula única
unique_grid = generate_unique_grid(env_config)

# Guardar la cuadrícula si es necesario (opcional)
# np.save("unique_grid.npy", unique_grid)

# Actualizar la configuración del entorno para incluir la cuadrícula
env_config["grid"] = unique_grid


# Iniciar el entrenamiento comparativo
def main():
    os.makedirs(
        "trained_agents", exist_ok=True
    )  # Asegurarse de que el directorio existe
    train_and_compare(
        env_config, agent_config_no_risk, agent_config_with_risk, train_config
    )


if __name__ == "__main__":
    main()
