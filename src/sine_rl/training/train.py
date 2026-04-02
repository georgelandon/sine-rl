import os
import argparse

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from gymnasium.wrappers.common import TimeLimit

from sine_rl.envs.sine_env import SineEnv
from sine_rl.utils.callbacks import make_eval_callback

def build_vec_env(episode_length: int, training: bool, log_dir: str, stacks: int, render_mode: str | None):
    env = DummyVecEnv([
        lambda: TimeLimit(
            SineEnv(episode_length=episode_length, training=training, render_mode=render_mode),
            episode_length
        )
    ])
    env = VecMonitor(env, log_dir)
    env = VecFrameStack(env, n_stack=stacks)
    return env

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--episode-length", type=int, default=600)
    p.add_argument("--stacks", type=int, default=8)

    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--eval-freq", type=int, default=512 * 100)
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--max-no-improvement-evals", type=int, default=10)

    p.add_argument("--runs-dir", type=str, default="runs")
    args = p.parse_args()

    train_log_dir = os.path.join(args.runs_dir, "train")
    eval_log_dir = os.path.join(args.runs_dir, "eval")
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    # No rendering in training for speed
    env = build_vec_env(args.episode_length, training=True, log_dir=train_log_dir, stacks=args.stacks, render_mode=True)
    # No window by default in eval; change to render_mode="human" if you want visible eval
    env_eval = build_vec_env(args.episode_length, training=False, log_dir=eval_log_dir, stacks=args.stacks, render_mode=True)

    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        stats_window_size=args.episode_length,
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
    )

    eval_cb = make_eval_callback(
        eval_env=env_eval,
        eval_log_dir=eval_log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        max_no_improvement_evals=args.max_no_improvement_evals,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=[eval_cb])

    # Save last model too (best model already saved by EvalCallback)
    model.save(os.path.join(eval_log_dir, "last_model"))

if __name__ == "__main__":
    main()
