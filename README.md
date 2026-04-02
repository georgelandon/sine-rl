# Sine RL (PPO + Gymnasium + Pygame)

Train a PPO agent (Stable-Baselines3) to track a sine wave by applying continuous actions that adjust `y_pred`.

Run commands from the repository root with the included launcher:

```powershell
python run.py --help
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

## Quick Start

Train with the default settings:

```powershell
python run.py train
```

Evaluate the saved best model and render a rollout:

```powershell
python run.py eval --rollout-steps 600 --render human
```

`eval` uses `runs/eval/best_model.zip` by default, so you only need `--model-path` if you want to load a different checkpoint.

## Example

This example keeps the run shorter and writes to a separate output folder:

```powershell
python run.py train --runs-dir runs/example --total-timesteps 20000 --episode-length 300
python run.py eval --runs-dir runs/example --rollout-steps 300 --render human
```

## Outputs

- `runs/train/monitor.csv`: training monitor logs
- `runs/eval/best_model.zip`: best checkpoint saved during evaluation
- `runs/eval/last_model.zip`: final checkpoint saved after training
- `runs/eval/evaluations.npz`: evaluation history used by Stable-Baselines3

## Help

```powershell
python run.py train --help
python run.py eval --help
```
