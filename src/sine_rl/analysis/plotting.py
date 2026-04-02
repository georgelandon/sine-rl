import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


def plot_eval_results(eval_log_dir: str, total_timesteps: int):
    results = load_results(eval_log_dir)
    x, y = ts2xy(results, "timesteps")
    plot_results([eval_log_dir], total_timesteps, results_plotter.X_TIMESTEPS, "Evaluation Results")
    plt.show()
    return x, y
