
from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("SAC/core/metadata.py", {})
DEFAULTS = SETTINGS["defaults"].get("SAC/core/metadata.py", {})
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import gymnasium
import motornet as mn
import numpy as np
import rich
import stable_baselines3
import torch

from .env import radius, sequence as default_sequence, target_hit_radius


def write_metadata(run_dir: Path, model, requested_timesteps: int, seed: int,
                   eval_freq: int, checkpoint_freq: int, eval_episodes: int,
                   actual_timesteps: int, early_stop_threshold, early_stop_patience,
                   stopped_early: bool, stop_reason: str,
                   straight_path_reward: bool = DEFAULTS["write_metadata"]["straight_path_reward"], *, algo: str = DEFAULTS["write_metadata"]["algo"],
                   sequence=default_sequence, hit_radius: float = target_hit_radius):
  if algo not in {"ppo", "sac"}:
    raise ValueError("algo must be 'ppo' or 'sac'")
  schedule_value = lambda schedule: float(schedule(1.0))
  data = {
    "algorithm": algo,
    "seed": seed, "requested_timesteps": requested_timesteps,
    "actual_timesteps": actual_timesteps,
    "sequence": list(sequence), "dt": 0.01,
    "reward": {"target": 20.0, "completion": 200.0, "progress_scale": 1.0,
               "straight_path": bool(straight_path_reward),
               "path_penalty_scale": 0.5 if straight_path_reward else 0.0,
               "travel_penalty_scale": 0.5 if straight_path_reward else 0.0},
    "environment": {"radius": radius, "target_hit_radius": hit_radius,
                    "tolerance": 0.01, "hold_steps": 20, "max_episode_steps": 800},
    "observation_shape": list(model.observation_space.shape),
    "action_shape": list(model.action_space.shape),
    "action_low": model.action_space.low.tolist(), "action_high": model.action_space.high.tolist(),
    "evaluation": {"eval_freq": eval_freq, "checkpoint_freq": checkpoint_freq,
                    "eval_episodes": eval_episodes, "deterministic": True},
    "early_stopping": {"threshold": early_stop_threshold, "patience": early_stop_patience,
                        "stopped_early": stopped_early, "reason": stop_reason},
    "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                  "numpy": np.__version__, "torch": torch.__version__, "gymnasium": gymnasium.__version__,
                  "motornet": mn.__version__, "stable_baselines3": stable_baselines3.__version__,
                  "rich": version("rich")},
  }
  if algo == "ppo":
    data["ppo"] = {
      "n_envs": 1, "n_steps": model.n_steps, "batch_size": model.batch_size,
      "n_epochs": model.n_epochs, "gamma": model.gamma, "gae_lambda": model.gae_lambda,
      "learning_rate": schedule_value(model.lr_schedule), "clip_range": schedule_value(model.clip_range),
      "loss": "clipped PPO policy loss + value loss + entropy regularization",
    }
  else:
    data["sac"] = {
      "n_envs": 1, "learning_rate": schedule_value(model.lr_schedule),
      "buffer_size": model.buffer_size, "learning_starts": model.learning_starts,
      "batch_size": model.batch_size, "gamma": model.gamma, "tau": model.tau,
      "train_freq": {"frequency": model.train_freq.frequency, "unit": model.train_freq.unit.value},
      "gradient_steps": model.gradient_steps, "ent_coef": model.ent_coef,
      "target_entropy": "auto", "target_entropy_value": model.target_entropy,
      "target_update_interval": model.target_update_interval,
      "network": "default SAC MLP", "replay_buffer_saved": False,
    }
  (run_dir / "config.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
