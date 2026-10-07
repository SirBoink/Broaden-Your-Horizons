from typing import Any

import gymnasium as gym
import numpy as np


def _squeeze(value: Any) -> Any:
  if isinstance(value, dict):
    return {key: _squeeze(item) for key, item in value.items()}
  if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 1:
    return value[0]
  return value


class SequentialReachGym(gym.Env):
  """Expose one batched MotorNet task as one Gymnasium environment."""

  metadata = {"render_modes": []}

  def __init__(self, task):
    self.task = task
    self.observation_space = task.observation_space
    self.action_space = task.action_space

  def reset(self, *, seed: int | None = None, options: dict | None = None):
    super().reset(seed=seed)
    options = {} if options is None else dict(options)
    options["batch_size"] = 1
    observation, info = self.task.reset(seed=seed, options=options)
    info = _squeeze(info)
    info["is_success"] = False
    return np.asarray(observation[0], dtype=np.float32), info

  def step(self, action):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != self.action_space.shape:
      raise ValueError(f"action shape {action.shape} != {self.action_space.shape}")
    observation, reward, terminated, truncated, info = self.task.step(action[None, :])
    info = _squeeze(info)
    info["is_success"] = bool(info["finished"])
    return (
      np.asarray(observation[0], dtype=np.float32),
      float(reward[0, 0]),
      bool(terminated),
      bool(truncated),
      info,
    )

  def close(self):
    return None
