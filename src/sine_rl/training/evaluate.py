import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from gymnasium.wrappers.common import TimeLimit

from sine_rl.envs.sine_env import SineEnv
from sine_rl.utils.plotting import plot_eval_results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--episode-length", type=int, default=600)
    p.add_argument("--stacks", type=int, default=8)

    p.add_argument("--eval-log-dir", type=str, default="runs/eval")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)

    # Optional: show window while stepping
    p.add_argument("--render", type=str, default="none", choices=["none", "human"])
    p.add_argument("--rollout-steps", type=int, default=0, help="If >0, run a rollout for this many steps.")
    args = p.parse_args()

    os.makedirs(args.eval_log_dir, exist_ok=True)

    render_mode = None if args.render == "none" else "human"

    env = DummyVecEnv([
        lambda: TimeLimit(
            SineEnv(episode_length=args.episode_length, training=False, render_mode=render_mode),
            args.episode_length
        )
    ])
    env = VecMonitor(env, args.eval_log_dir)
    env = VecFrameStack(env, n_stack=args.stacks)

    model = PPO.load(args.model_path, env=env)

    if args.rollout_steps > 0:
        obs = env.reset()
        for _ in range(args.rollout_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

    plot_eval_results(args.eval_log_dir, args.total_timesteps)

if __name__ == "__main__":
    main()
