import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json


def plot_episode_metric(metric_data, metric_name, save_path):
    """Create a line plot for a specific metric across episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(metric_data, marker="o", linestyle="-", markersize=4)
    plt.title(f"{metric_name} per Episode")
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.grid(True)

    if save_path:
        plt.savefig(
            os.path.join(
                save_path, f'{metric_name.lower().replace(" ", "_")}_per_episode.png'
            )
        )
        plt.close()
    else:
        plt.show()


def get_metrics_from_history(history):
    """Extract all available metrics from the history."""
    if not history:
        return {}

    metrics = {}

    # For list of dictionaries format
    if isinstance(history, list) and len(history) > 0:
        # Get all unique keys from all episodes
        all_keys = set()
        for episode in history:
            all_keys.update(episode.keys())

        # Create metrics for each key
        for key in all_keys:
            try:
                if key == "success":
                    # Convert boolean success to 0/1 for visualization
                    metrics[f"Success Rate"] = [
                        1 if h.get(key, False) else 0 for h in history
                    ]
                else:
                    # Convert key to title case for better display
                    metric_name = " ".join(word.capitalize() for word in key.split("_"))
                    metrics[metric_name] = [h.get(key, 0) for h in history]
            except Exception as e:
                print(f"Warning: Could not process metric {key}: {str(e)}")

    # For dictionary of lists format
    elif isinstance(history, dict):
        for key, values in history.items():
            try:
                if key == "success":
                    # Convert boolean success to 0/1 for visualization
                    metrics["Success Rate"] = [1 if s else 0 for s in values]
                else:
                    # Convert key to title case for better display
                    metric_name = " ".join(word.capitalize() for word in key.split("_"))
                    metrics[metric_name] = values
            except Exception as e:
                print(f"Warning: Could not process metric {key}: {str(e)}")

    return metrics


def create_episode_graphs(history, save_path=None):
    """Create individual graphs for each metric in the history."""
    # Create directory if saving
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Get metrics dynamically from history
    metrics = get_metrics_from_history(history)

    if not metrics:
        print("Warning: No metrics found in history")
        return

    print(f"Generating graphs for metrics: {', '.join(metrics.keys())}")

    # Plot each metric
    for metric_name, metric_data in metrics.items():
        try:
            plot_episode_metric(metric_data, metric_name, save_path)
        except Exception as e:
            print(f"Error plotting metric {metric_name}: {str(e)}")


def load_history(filepath):
    """Load history from a pickle file."""
    if filepath.endswith(".pickle"):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif filepath.endswith(".json"):
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .pickle or .json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate episode-by-episode graphs")
    parser.add_argument(
        "history_path", type=str, help="Path to the history file (.pickle or .json)"
    )
    parser.add_argument(
        "--save_dir", type=str, help="Directory to save the graphs", default=None
    )

    args = parser.parse_args()
    history = load_history(args.history_path)
    create_episode_graphs(history, args.save_dir)
