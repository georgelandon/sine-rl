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
        "--eval-log-dir",
        type=str,
        default=None,
        help="Directory that stores evaluation logs. Defaults to <runs-dir>/eval.",
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)

    parser.add_argument("--render", type=str, default="none", choices=["none", "human"])
    parser.add_argument("--rollout-steps", type=int, default=0, help="If >0, run a rollout for this many steps.")
    return parser

def run(args: argparse.Namespace) -> int:
    from gymnasium.wrappers.common import TimeLimit
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

    from sine_rl.envs.sine_env import SineEnv
    from sine_rl.utils.plotting import plot_eval_results

    eval_log_dir = args.eval_log_dir or os.path.join(args.runs_dir, "eval")
    model_path = args.model_path or os.path.join(eval_log_dir, "best_model.zip")

    os.makedirs(eval_log_dir, exist_ok=True)

    render_mode = None if args.render == "none" else "human"

    env = DummyVecEnv([
        lambda: TimeLimit(
            SineEnv(episode_length=args.episode_length, training=False, render_mode=render_mode),
            args.episode_length
        )
    ])
    env = VecMonitor(env, eval_log_dir)
    env = VecFrameStack(env, n_stack=args.stacks)

    try:
        model = PPO.load(model_path, env=env)

        if args.rollout_steps > 0:
            obs = env.reset()
            for _ in range(args.rollout_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                if done:
                    obs = env.reset()

        plot_eval_results(eval_log_dir, args.total_timesteps)
    finally:
        env.close()

    return 0

def main(argv=None) -> int:
    parser = configure_parser(
        argparse.ArgumentParser(description="Evaluate a saved PPO sine-wave model.")
    )
    return run(parser.parse_args(argv))

if __name__ == "__main__":
    raise SystemExit(main())
