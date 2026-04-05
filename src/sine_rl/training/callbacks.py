from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnNoModelImprovement


class RenderTestAfterEvalCallback(BaseCallback):
    def __init__(
        self,
        episode_length: int,
        stacks: int,
        render_mode: str | None,
        rollout_steps: int = 0,
        deterministic: bool = True,
    ):
        super().__init__()
        self.episode_length = episode_length
        self.stacks = stacks
        self.render_mode = render_mode
        self.rollout_steps = rollout_steps if rollout_steps > 0 else episode_length
        self.deterministic = deterministic
        self.render_env = None

    def _init_callback(self) -> None:
        if self.render_mode is None:
            return

        from gymnasium.wrappers.common import TimeLimit
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

        from sine_rl.envs.sine_env import SineEnv

        self.render_env = DummyVecEnv([
            lambda: TimeLimit(
                SineEnv(
                    episode_length=self.episode_length,
                    training=False,
                    render_mode=self.render_mode,
                ),
                self.episode_length,
            )
        ])
        self.render_env = VecFrameStack(self.render_env, n_stack=self.stacks)

    def _on_step(self) -> bool:
        if self.render_env is None:
            return True

        obs = self.render_env.reset()
        for _ in range(self.rollout_steps):
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, done, _ = self.render_env.step(action)
            if done:
                obs = self.render_env.reset()

        return True

    def _on_training_end(self) -> None:
        if self.render_env is not None:
            self.render_env.close()

def make_eval_callback(
    eval_env,
    eval_log_dir: str,
    eval_freq: int,
    n_eval_episodes: int = 20,
    max_no_improvement_evals: int = 10,
    deterministic: bool = True,
    render_mode: str | None = None,
    render_rollout_steps: int = 0,
    episode_length: int = 600,
    stacks: int = 8,
):
    stopper = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=max_no_improvement_evals,
        verbose=1,
    )

    callbacks = [stopper]

    if render_mode is not None:
        callbacks.insert(
            0,
            RenderTestAfterEvalCallback(
                episode_length=episode_length,
                stacks=stacks,
                render_mode=render_mode,
                rollout_steps=render_rollout_steps,
                deterministic=deterministic,
            ),
        )

    callback_after_eval = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

    return EvalCallback(
        eval_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=eval_freq,
        deterministic=deterministic,
        n_eval_episodes=n_eval_episodes,
        callback_after_eval=callback_after_eval,
    )
