import os
import argparse

def build_vec_env(episode_length: int, training: bool, log_dir: str, stacks: int, render_mode: str | None):
    from gymnasium.wrappers.common import TimeLimit
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

    from sine_rl.envs.sine_env import SineEnv

    env = DummyVecEnv([
        lambda: TimeLimit(
            SineEnv(episode_length=episode_length, training=training, render_mode=render_mode),
            episode_length
        )
    ])
    env = VecMonitor(env, log_dir)
    env = VecFrameStack(env, n_stack=stacks)
    return env

def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--episode-length", type=int, default=600)
    parser.add_argument("--stacks", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--eval-freq", type=int, default=512 * 100)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--max-no-improvement-evals", type=int, default=10)

    parser.add_argument("--runs-dir", type=str, default="runs")
    return parser

def run(args: argparse.Namespace) -> int:
    import torch.nn as nn
    from stable_baselines3 import PPO

    from sine_rl.utils.callbacks import make_eval_callback

    train_log_dir = os.path.join(args.runs_dir, "train")
    eval_log_dir = os.path.join(args.runs_dir, "eval")
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    env = build_vec_env(
        args.episode_length,
        training=True,
        log_dir=train_log_dir,
        stacks=args.stacks,
        render_mode=None,
    )
    env_eval = build_vec_env(
        args.episode_length,
        training=False,
        log_dir=eval_log_dir,
        stacks=args.stacks,
        render_mode=None,
    )

    try:
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

        model.save(os.path.join(eval_log_dir, "last_model"))
    finally:
        env.close()
        env_eval.close()

    return 0

def main(argv=None) -> int:
    parser = configure_parser(
        argparse.ArgumentParser(description="Train a PPO agent to track a sine wave.")
    )
    return run(parser.parse_args(argv))

if __name__ == "__main__":
    raise SystemExit(main())
