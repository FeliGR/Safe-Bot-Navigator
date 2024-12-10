import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class BasicQLearningAgent:

    TRAIN_HISTORY = "train"
    GREEDY_HISTORY = "greedy"
    BOTH_HISTORIES = "both"
    VALID_HISTORY_TYPES = {TRAIN_HISTORY, GREEDY_HISTORY, BOTH_HISTORIES}

    def __init__(
        self,
        learning_rate,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        xi,
        state_size,
        n_actions,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.xi = xi
        self.state_size = state_size
        self.n_actions = n_actions
        self.q_table = np.zeros((state_size, n_actions))
        self.q_risk = np.zeros((state_size, n_actions))
        self.train_history = {
            "episodes": [],
            "rewards": [],
            "risk": [],
            "steps": [],
            "max_q": [],
            "max_q_risk": [],
        }
        self.greedy_history = {}

    def get_action(self, state):
        """Elegir acción usando una política epsilon-greedy basada en Qξ"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_xi = self.xi * self.q_table[state] - self.q_risk[state]
        options = np.argwhere(q_xi == np.max(q_xi)).flatten()
        return np.random.choice(options)

    def update(self, state, action, reward, next_state, risk_reward):
        """Actualizar Q y Q_risk para el par estado-acción"""
        # Actualización de Q
        current_q = self.q_table[state][action]
        future_q = np.max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.gamma * future_q - current_q
        )

        # Actualización de Q_risk
        current_q_risk = self.q_risk[state][action]
        future_q_risk = np.max(self.q_risk[next_state])
        self.q_risk[state][action] = current_q_risk + self.learning_rate * (
            risk_reward + self.gamma * future_q_risk - current_q_risk
        )

        # Actualización de Qξ si es necesario

    def decay_epsilon(self):
        """Decay epsilon after an episode by subtracting epsilon_decay"""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def train(
        self,
        env,
        episodes=10000,
        max_steps=1000,
        render_freq=100,
        render_mode=None,
        render_delay=0.1,
    ):
        """Entrenar el agente usando Q-Learning con sensibilidad al riesgo"""

        history = {
            "rewards": [],
            "epsilon": [],
            "max_q": [],
            "max_q_risk": [],
            "steps": [],
            "episodes": list(range(episodes)),
        }

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            total_risk = 0
            steps = 0
            done = False

            while not done and steps < max_steps:
                if render_mode and episode % render_freq == 0:
                    env.render(mode=render_mode)
                    time.sleep(render_delay)

                action = self.get_action(state)
                next_state, reward, done, is_error = env.step(action)

                risk_reward = 1 if is_error else 0
                self.update(state, action, reward, next_state, risk_reward)

                state = next_state
                total_reward += reward
                total_risk += risk_reward
                steps += 1

            if render_mode and episode % render_freq == 0:
                env.render(mode=render_mode)
                time.sleep(render_delay)

            history["rewards"].append(total_reward)
            history["epsilon"].append(self.epsilon)
            history["max_q"].append(np.max(self.q_table))
            history["max_q_risk"].append(np.max(self.q_risk))
            history["steps"].append(steps)

            if episode % 100 == 0:
                print(
                    f"Episode {episode}/{episodes}, Steps: {steps}, "
                    f"Reward: {total_reward:.2f}, Risk: {total_risk}, "
                    f"Epsilon: {self.epsilon:.2f}, Max Q: {np.max(self.q_table):.2f}, "
                    f"Max Q_risk: {np.max(self.q_risk):.2f}"
                )

            self.decay_epsilon()

        if render_mode:
            env.close()

        self.train_history = history

    def run_greedy(
        self, env, episodes=1, max_steps=1000, render_mode="human", render_delay=0.1
    ):
        """Ejecutar el agente usando una política greedy (sin exploración) para múltiples episodios.
        
        Args:
            env: El entorno en el que ejecutar
            episodes: Número de episodios a ejecutar
            max_steps: Número máximo de pasos por episodio
            render_mode: Cómo renderizar el entorno (None o 'human')
            render_delay: Retraso entre renders en segundos
        
        Returns:
            dict: Historial que contiene recompensas, pasos y tasa de éxito
        """
        history = {
            "rewards": [],
            "steps": [],
            "success": [],
            "episodes": list(range(episodes)),
        }

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and steps < max_steps:
                if render_mode:
                    env.render(mode=render_mode)
                    time.sleep(render_delay)

                action = np.argmax(self.q_table[state])
                next_state, reward, done, is_error = env.step(action)

                state = next_state
                total_reward += reward

                if is_error:
                    history["success"].append(True)
                    done = True
                else:
                    history["success"].append(done)

                steps += 1

            if render_mode:
                env.render(mode=render_mode)
                time.sleep(render_delay)

            history["rewards"].append(total_reward)
            history["steps"].append(steps)

            if episode % 10 == 0:
                success_rate = sum(history["success"][-10:]) / min(10, episode + 1)
                print(
                    f"Episode {episode}/{episodes}, Steps: {steps}, "
                    f"Reward: {total_reward:.2f}, Success Rate: {success_rate:.2f}, "
                    f"Epsilon: {self.epsilon:.2f}"
                )

        if render_mode:
            env.close()

        self.greedy_history = history

    def plot_history(self, history_type=BOTH_HISTORIES, save_path=None):
        """Plot agent histories in a grid layout.

        Args:
            history_type: Which history to plot (TRAIN_HISTORY, GREEDY_HISTORY, or BOTH_HISTORIES)
            save_path: Optional path to save the plot
        """
        if history_type not in self.VALID_HISTORY_TYPES:
            raise ValueError(
                f"Invalid history type. Must be one of {self.VALID_HISTORY_TYPES}"
            )

        histories = []
        if (
            history_type in [self.TRAIN_HISTORY, self.BOTH_HISTORIES]
            and self.train_history
        ):
            histories.append(("Training", self.train_history))
        if (
            history_type in [self.GREEDY_HISTORY, self.BOTH_HISTORIES]
            and self.greedy_history
        ):
            histories.append(("Greedy", self.greedy_history))

        if not histories:
            print("No history available to plot")
            return

        # Calculate total number of metrics to plot
        n_metrics = sum(
            len([k for k in h[1].keys() if k != "episodes"]) for h in histories
        )

        # Calculate grid dimensions
        n_cols = min(2, n_metrics)  # Maximum 2 columns
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

        # Create figure with appropriate size
        plt.figure(figsize=(8 * n_cols, 5 * n_rows))

        plot_idx = 1

        for title, history in histories:
            metrics = [k for k in history.keys() if k != "episodes"]
            episodes = history["episodes"]

            for metric in metrics:
                plt.subplot(n_rows, n_cols, plot_idx)
                plt.plot(episodes, history[metric], label=f"{title} {metric}")
                plt.title(f'{title} {metric.replace("_", " ").title()}')
                plt.xlabel("Episodes")
                plt.ylabel(metric.replace("_", " ").title())
                plt.grid(True)
                plt.legend()

                plot_idx += 1

        plt.tight_layout()  # Adjust spacing between subplots

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        plt.close()

    def adjust_xi(self, increment=0.01, max_xi=10.0):
        """Incrementar ξ hasta alcanzar el máximo permitido"""
        if self.xi < max_xi:
            self.xi += increment
            print(f"ξ incrementado a {self.xi:.2f}")

    def calculate_risk_map(self, env):
        """Calcula el riesgo promedio por celda basado en Q_risk."""
        # Inicializar como arreglos 2D
        risk_map = np.zeros((env.size, env.size))
        count = np.zeros((env.size, env.size))

        for state in range(self.state_size):
            i, j = divmod(state, env.size)
            for action in range(self.n_actions):
                risk_map[i, j] += self.q_risk[state, action]
                count[i, j] += 1

        # Evitar división por cero
        count[count == 0] = 1
        risk_map /= count
        return risk_map

    def plot_risk_map(self, env, save_path=None):
        """Genera y muestra un mapa de calor del riesgo."""
        risk_map = self.calculate_risk_map(env)
        plt.figure(figsize=(8, 6))
        sns.heatmap(risk_map, annot=True, cmap="Reds", fmt=".2f", cbar=True)
        plt.title("Mapa de Riesgo")
        plt.xlabel("Columna")
        plt.ylabel("Fila")
        plt.gca().invert_yaxis()  # Para que la fila 0 esté en la parte superior

        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()

def plot_comparison(agent_no_risk, agent_with_risk, window=100):
    """Genera gráficos comparativos entre dos agentes con suavizado en rewards.
    
    Args:
        agent_no_risk: Agente sin sensibilidad al riesgo.
        agent_with_risk: Agente con sensibilidad al riesgo.
        window: Tamaño de la ventana para la media móvil.
    """
    metrics = ["rewards", "steps"]
    episodes = agent_no_risk.train_history["episodes"]

    plt.figure(figsize=(18, 5))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        if metric == "rewards":
            # Convertir las recompensas a Series de pandas para aplicar la media móvil
            rewards_no_risk = pd.Series(agent_no_risk.train_history[metric])
            rewards_with_risk = pd.Series(agent_with_risk.train_history[metric])

            # Calcular la media móvil
            rewards_no_risk_smooth = rewards_no_risk.rolling(window=window).mean()
            rewards_with_risk_smooth = rewards_with_risk.rolling(window=window).mean()

            plt.plot(episodes, rewards_no_risk_smooth, label="Sin Riesgo (Suavizado)")
            plt.plot(episodes, rewards_with_risk_smooth, label="Con Riesgo (Suavizado)")

            # Opcional: También puedes mostrar las recompensas originales con transparencia
            plt.plot(episodes, rewards_no_risk, color='blue', alpha=0.1)
            plt.plot(episodes, rewards_with_risk, color='orange', alpha=0.1)
        else:
            plt.plot(episodes, agent_no_risk.train_history[metric], label="Sin Riesgo")
            plt.plot(episodes, agent_with_risk.train_history[metric], label="Con Riesgo")

        plt.title(f"Comparación de {metric.capitalize()}")
        plt.xlabel("Episodios")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
