# Project Name

## Project Overview
This project involves implementing and training various learning agents in a configurable grid environment. The agents learn to navigate, avoid risks, and achieve goals using diverse strategies. The project includes tools for visualization and performance analysis.

---

## Installation Steps for Mac and Windows

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Conda (Download from [Anaconda](https://www.anaconda.com/) for macOS and Windows)

### Steps for macOS and Windows
1. Clone the repository:
   ```bash
   git clone https://github.com/FeliGR/Safe-Bot-Navigator.git
   cd Safe-Bot-Navigator
   ```
2. Create a Conda environment and activate it:
   ```bash
   conda create --name myenv python=3.x
   conda activate myenv
   ```
3. Install dependencies:
   ```bash
   pip install numpy matplotlib pygame pickle-mixin jsonschema
   ```

---

## Project Structure
- **agents**: Implementation of different learning agents.
- **environment**: Grid environment implementation.
- **execution**: Scripts for running trained agents.
- **training**: Training configurations and scripts.
- **visualization**: Tools for visualizing agent performance.
- **safe_results**: Directory for storing training results.
- **trained_agents**: Directory for storing trained agent models.

---

## Usage Guide

### Training an Agent
1. Configure training parameters in the `config` file.
2. Run the training script:
   ```bash
   python training/train_agent.py
   ```

### Running a Trained Agent
1. Use the execution script to run a trained agent:
   ```bash
   python execution/run_agent.py
   ```

### Visualizing Results
1. Use visualization tools to analyze agent performance:
   ```bash
   python visualization/plot_results.py
   ```

---

## Environment

The grid environment consists of:
- **Empty cells**: Traversable spaces for the agent.
- **Obstacles**: Non-traversable barriers.
- **Traps**: Hazardous areas with penalties.
- **Target**: The goal position for the agent.
- **Robot**: The agent navigating the environment.

---

## Configuration
Key parameters that can be configured:
- **Grid size**
- **Obstacle probability**
- **Trap probability**
- **Trap danger level**
- **Rewards** for different events
- **Learning parameters**: Learning rate, discount factor, etc.
- **Training episodes** and steps

---

## Features
- Multiple agent types with different learning strategies
- Risk-aware navigation
- Teacher-guided learning
- Performance visualization
- Configurable environment parameters
- Save/load trained agents
- Episode metrics tracking and visualization

---

## Dependencies

The following dependencies are required:
- Python 3.x
- `NumPy`
- `Matplotlib`
- `Pygame`
- `Pickle`
- `JSON`

Install dependencies with:
```bash
pip install numpy matplotlib pygame pickle-mixin jsonschema
```

---