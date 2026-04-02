import argparse

from sine_rl.training import evaluate, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the sine-wave PPO agent from the repository root."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a PPO agent.")
    train.configure_parser(train_parser)
    train_parser.set_defaults(handler=train.run)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved PPO agent.")
    evaluate.configure_parser(eval_parser)
    eval_parser.set_defaults(handler=evaluate.run)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
