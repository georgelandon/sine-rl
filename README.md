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

Train with the default settings:

```powershell
python scripts/train.py
```

Render training live in a Pygame window:

```powershell
python scripts/train.py --render human
```

Evaluate the saved best model and render a rollout:

```powershell
python scripts/eval.py --rollout-steps 600 --render human
```

`eval` uses `runs/eval/best_model.zip` by default, so you only need `--model-path` if you want to load a different checkpoint.

## Example

This example keeps the run shorter and writes to a separate output folder:

```powershell
python scripts/train.py --runs-dir runs/example --total-timesteps 20000 --episode-length 300
python scripts/eval.py --runs-dir runs/example --rollout-steps 300 --render human
```

If you want to watch learning happen, add `--render human` to the training command. It is useful for debugging, but it will make training slower.

## Outputs

- `runs/train/monitor.csv`: training monitor logs
- `runs/eval/best_model.zip`: best checkpoint saved during evaluation
- `runs/eval/last_model.zip`: final checkpoint saved after training
- `runs/eval/evaluations.npz`: evaluation history used by Stable-Baselines3

Outputs live under `runs/` at the repo root and are intentionally kept out of `src/`.

## Help

```powershell
python scripts/train.py --help
python scripts/eval.py --help
```

After installation, you can also use the console scripts:

```powershell
sine-rl-train --help
sine-rl-eval --help
```
