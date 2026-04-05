# Sine RL (PPO + Gymnasium + Pygame)

Train a PPO agent (Stable-Baselines3) to track a sine wave by applying continuous actions that adjust `y_pred`.

This project now follows a more typical RL layout:

```text
sine-rl/
|-- scripts/              # runnable entrypoints
|-- src/sine_rl/envs/     # Gymnasium environments
|-- src/sine_rl/training/ # PPO training and evaluation code
|-- src/sine_rl/analysis/ # plots and result inspection
|-- runs/                 # outputs (gitignored)
```

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the package in editable mode, so imports work cleanly with the `src/` layout.

## Quick Start

Train with the default settings. Training stays headless for speed:

```powershell
python scripts/train.py
```

The training defaults now match the old reference script more closely:
- default SB3 `MlpPolicy`
- `--train-steps 100000000`
- `--episode-length 600`
- `--stacks 8`
- `--learning-rate 0.0001`
- `--n-steps 256`
- `--batch-size 64`
- `--eval-freq 51200`
- `--n-eval-episodes 20`
- `--max-no-improvement-evals 10`

Use `--total-timesteps` (or `--train-steps`) to make training run longer. `--n-steps` only controls how many steps PPO collects before each update.

Render a test rollout after each scheduled evaluation during training:

```powershell
python scripts/train.py --eval-render human
```

In this project, the closest equivalent to "after each epoch" is "after each scheduled evaluation". That cadence is controlled by `--eval-freq`.

Test the saved best model with a rendered rollout:

```powershell
python scripts/test.py
```

If you want a non-rendered test run:

```powershell
python scripts/test.py --render none
```

`test` uses `runs/eval/best_model.zip` by default, so you only need `--model-path` if you want to load a different checkpoint.
Training still writes validation outputs to `runs/eval/` during learning so the best checkpoint can be selected automatically.

## Example

This example keeps the run shorter and writes to a separate output folder:

```powershell
python scripts/train.py --runs-dir runs/example --total-timesteps 20000 --episode-length 300 --eval-freq 5000 --eval-render human
python scripts/test.py --runs-dir runs/example --rollout-steps 300
```

Reference-like training call:

```powershell
python scripts/train.py --train-steps 100000000 --episode-length 600 --stacks 8 --learning-rate 0.0001 --n-steps 256 --batch-size 64 --eval-freq 51200 --n-eval-episodes 20 --max-no-improvement-evals 10 --eval-render none
```

`scripts/eval.py` is still available as a lower-level evaluation utility, but the normal workflow is now `train` then `test`.

## Outputs

- `runs/train/monitor.csv`: training monitor logs
- `runs/eval/best_model.zip`: best checkpoint selected during training-time evaluation
- `runs/eval/last_model.zip`: final checkpoint saved after training
- `runs/eval/evaluations.npz`: training-time evaluation history used by Stable-Baselines3
- `runs/test/monitor.csv`: post-training test rollout logs

Outputs live under `runs/` at the repo root and are intentionally kept out of `src/`.

## Help

```powershell
python scripts/train.py --help
python scripts/test.py --help
python scripts/eval.py --help
```

After installation, you can also use the console scripts:

```powershell
sine-rl-train --help
sine-rl-test --help
sine-rl-eval --help
```
