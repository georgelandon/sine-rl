from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

def make_eval_callback(
    eval_env,
    eval_log_dir: str,
    eval_freq: int,
    n_eval_episodes: int = 20,
    max_no_improvement_evals: int = 10,
    deterministic: bool = True,
):
    stopper = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=max_no_improvement_evals,
        verbose=1,
    )

    return EvalCallback(
        eval_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=eval_freq,
        deterministic=deterministic,
        n_eval_episodes=n_eval_episodes,
        callback_after_eval=stopper,
    )
