import os
import argparse


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a saved model. Defaults to <runs-dir>/eval/best_model.zip.",
    )
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--episode-length", type=int, default=600)
    parser.add_argument("--stacks", type=int, default=8)
    parser.add_argument(
        "--test-log-dir",
        type=str,
        default=None,
        help="Directory that stores test logs. Defaults to <runs-dir>/test.",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument(
        "--render",
        type=str,
        default="human",
        choices=["none", "human"],
        help="Render the test rollout in a Pygame window.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=0,
        help="Number of test steps to run. Defaults to one full episode when omitted.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    from gymnasium.wrappers.common import TimeLimit
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

    from sine_rl.analysis.plotting import plot_results_dir
    from sine_rl.envs.sine_env import SineEnv

    test_log_dir = args.test_log_dir or os.path.join(args.runs_dir, "test")
    model_path = args.model_path or os.path.join(args.runs_dir, "eval", "best_model.zip")
    rollout_steps = args.rollout_steps if args.rollout_steps > 0 else args.episode_length

    os.makedirs(test_log_dir, exist_ok=True)

    render_mode = None if args.render == "none" else "human"

    env = DummyVecEnv([
        lambda: TimeLimit(
            SineEnv(episode_length=args.episode_length, training=False, render_mode=render_mode),
            args.episode_length
        )
    ])
    env = VecMonitor(env, test_log_dir)
    env = VecFrameStack(env, n_stack=args.stacks)

    try:
        model = PPO.load(model_path, env=env)

        obs = env.reset()
        for _ in range(rollout_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

        plot_results_dir(test_log_dir, args.total_timesteps, title="Test Results")
    finally:
        env.close()

    return 0


def main(argv=None) -> int:
    parser = configure_parser(
        argparse.ArgumentParser(description="Test a saved PPO sine-wave model with an optional rendered rollout.")
    )
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
