import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


def plot_results_dir(log_dir: str, total_timesteps: int, title: str = "Results"):
    results = load_results(log_dir)
    x, y = ts2xy(results, "timesteps")
    plot_results([log_dir], total_timesteps, results_plotter.X_TIMESTEPS, title)
    plt.show()
    return x, y


def plot_eval_results(eval_log_dir: str, total_timesteps: int):
    return plot_results_dir(eval_log_dir, total_timesteps, title="Evaluation Results")
