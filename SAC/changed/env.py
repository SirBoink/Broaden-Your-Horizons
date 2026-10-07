
from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("SAC/changed/env.py", {})
DEFAULTS = SETTINGS["defaults"].get("SAC/changed/env.py", {})
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from SAC.core.env import SequentialReachEnv, radius
from SAC.core.ppo_env import SequentialReachGym

sequence = tuple(PARAMS["sequence"])


class StraightReachEnv(SequentialReachEnv):
  """Sequential reach task with learnable straight-path shaping."""

  def __init__(self, *args, distance_scale: float = DEFAULTS["StraightReachEnv.__init__"]["distance_scale"],
               path_penalty_scale: float = DEFAULTS["StraightReachEnv.__init__"]["path_penalty_scale"], arrival_speed: float = DEFAULTS["StraightReachEnv.__init__"]["arrival_speed"],
               random_start: bool = DEFAULTS["StraightReachEnv.__init__"]["random_start"], shuffle_sequence: bool = DEFAULTS["StraightReachEnv.__init__"]["shuffle_sequence"],
               start_pos=None, **kwargs):
    if any(not np.isfinite(value) or value < 0 for value in (
      distance_scale, path_penalty_scale, arrival_speed,
    )):
      raise ValueError("reward scales and arrival_speed must be finite and non-negative")
    self.distance_scale = float(distance_scale)
    self.path_penalty_scale = float(path_penalty_scale)
    self.arrival_speed = float(arrival_speed)
    self.random_start = bool(random_start)
    self.shuffle_sequence = bool(shuffle_sequence)
    if start_pos is not None:
      start_pos = np.asarray(start_pos, dtype=np.float32)
      if start_pos.shape != (2,) or not np.all(np.isfinite(start_pos)):
        raise ValueError("start_pos must contain two finite joint angles in degrees")
      start_pos = np.deg2rad(start_pos)
    self.start_pos = start_pos
    kwargs["max_episode_steps"] = kwargs.get("max_episode_steps", 800)
    kwargs["straight_path_reward"] = False
    super().__init__(*args, **kwargs)

  def _get_obs_size(self) -> int:
    return super()._get_obs_size() + self.effector.n_muscles

  def get_obs(self, action=None, deterministic: bool = DEFAULTS["StraightReachEnv.get_obs"]["deterministic"]):
    self.update_obs_buffer(action=action)
    physical = torch.cat([
      self.goal,
      self.obs_buffer["vision"][0],
      self.obs_buffer["proprioception"][0],
    ] + self.obs_buffer["action"][:self.action_frame_stacking], dim=-1)
    activation = self.states["muscle"][:, 0, :]
    if not deterministic:
      action_features = self.effector.n_muscles * self.action_frame_stacking
      physical = self.apply_noise(
        physical,
        noise=self.obs_noise[:2] + self.obs_noise[10:24 + action_features],
      )
      activation = self.apply_noise(
        activation, noise=self.obs_noise[24 + action_features:30 + action_features]
      )
    target = self._sequence[self.sequence_index]
    one_hot = torch.nn.functional.one_hot(
      target, num_classes=len(self.target_names)
    ).to(torch.float32)
    obs = torch.cat((physical[:, :2], one_hot, physical[:, 2:], activation), dim=-1)
    return obs if self.differentiable else self.detach(obs)

  def reset(self, *, seed: int | None = None,
            options: dict[str, Any] | None = None):
    self._set_generator(seed=seed)
    options = {} if options is None else dict(options)
    if self.shuffle_sequence:
      options["sequence"] = self.np_random.permutation(
        options.get("sequence", self.default_sequence)
      ).tolist()

    reference_posture = self.reference_posture
    if self.start_pos is not None:
      self.reference_posture = self.start_pos
    elif self.random_start:
      first_target = int(options.get("sequence", self.default_sequence)[0])
      choices = [target for target in range(len(self.target_names)) if target != first_target]
      start_target = int(self.np_random.choice(choices))
      self.reference_posture = self.detach(self.target_joint_angles[start_target])
    try:
      return super().reset(seed=None, options=options)
    finally:
      self.reference_posture = reference_posture

  def step(self, action: torch.Tensor | np.ndarray, deterministic: bool = DEFAULTS["StraightReachEnv.step"]["deterministic"],
           **kwargs):
    self.elapsed_steps += 1
    self.elapsed = self.elapsed_steps * self.dt
    action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
    noisy_action = action if deterministic else self.apply_noise(action, self.action_noise)
    evaluated_goal = self.goal.clone()
    evaluated_target = self._sequence[self.sequence_index].clone()
    active = ~self.finished
    before_hand = self.states["fingertip"].clone()
    before_distance = torch.linalg.vector_norm(before_hand - evaluated_goal, dim=1)
    self.effector.step(noisy_action, **kwargs)

    after_hand = self.states["fingertip"]
    evaluated_error = after_hand - evaluated_goal
    after_distance = torch.linalg.vector_norm(evaluated_error, dim=1)
    progress_dist = before_distance - after_distance
    progress_reward = self.progress_reward_scale * progress_dist / self.tolerance
    step_dist = torch.linalg.vector_norm(after_hand - before_hand, dim=1)
    wasted_dist = torch.clamp(step_dist - progress_dist, min=0.0)
    path_penalty = self.path_penalty_scale * wasted_dist / self.tolerance
    distance_reward = -self.distance_scale * after_distance / radius
    progress_reward = torch.where(active, progress_reward, torch.zeros_like(progress_reward))
    path_penalty = torch.where(active, path_penalty, torch.zeros_like(path_penalty))
    distance_reward = torch.where(active, distance_reward, torch.zeros_like(distance_reward))
    speed = step_dist / self.dt

    within = self._target_reached(after_distance, speed) & active
    if self.dwell:
      self.hold_count = torch.where(within, self.hold_count + 1, 0)
      completed = (self.hold_count >= self.hold_steps) & active
    else:
      completed = within
    last = self.sequence_index == len(self.sequence) - 1
    self.finished |= completed & last
    self.sequence_index += (completed & ~last).long()
    self.hold_count = torch.where(completed, 0, self.hold_count)
    self.goal = self.targets[self._sequence[self.sequence_index]]

    obs = self.get_obs(action=noisy_action, deterministic=deterministic)
    target_reward = SETTINGS["reward"]["target_acquisition"] * completed.float()
    completion_reward = SETTINGS["reward"]["sequence_completion"] * (completed & last).float()
    reward_tensor = (
      progress_reward + distance_reward - path_penalty
      + target_reward + completion_reward
    )
    timeout = self.elapsed_steps >= self.max_episode_steps
    terminated = bool(torch.all(self.finished).item())
    truncated = bool(timeout and not terminated)
    reward = None if self.differentiable else self.detach(reward_tensor[:, None])
    info = {
      "states": self._maybe_detach_states(),
      "action": action if self.differentiable else self.detach(action),
      "noisy action": noisy_action if self.differentiable else self.detach(noisy_action),
      "evaluated_goal": evaluated_goal if self.differentiable else self.detach(evaluated_goal),
      "evaluated_target": evaluated_target if self.differentiable else self.detach(evaluated_target),
      "evaluated_target_error": evaluated_error if self.differentiable else self.detach(evaluated_error),
      "distance_before": before_distance if self.differentiable else self.detach(before_distance),
      "distance_after": after_distance if self.differentiable else self.detach(after_distance),
      "progress_dist": progress_dist if self.differentiable else self.detach(progress_dist),
      "step_dist": step_dist if self.differentiable else self.detach(step_dist),
      "wasted_dist": wasted_dist if self.differentiable else self.detach(wasted_dist),
      "progress_reward": progress_reward if self.differentiable else self.detach(progress_reward),
      "distance_reward": distance_reward if self.differentiable else self.detach(distance_reward),
      "path_penalty": path_penalty if self.differentiable else self.detach(path_penalty),
      "travel": speed if self.differentiable else self.detach(speed),
      "travel_reward": -path_penalty if self.differentiable else self.detach(-path_penalty),
      "target_reward": target_reward if self.differentiable else self.detach(target_reward),
      "completion_reward": completion_reward if self.differentiable else self.detach(completion_reward),
      "completed": completed if self.differentiable else self.detach(completed),
      "timeout": timeout,
      **self._task_info(),
    }
    return obs, reward, terminated, truncated, info

  def _target_reached(self, distance, speed):
    return (distance <= self.hit_radius) & (speed <= self.arrival_speed)


class StraightReachGym(SequentialReachGym):
  """Expose normalized policy actions while MotorNet receives activations in [0, 1]."""

  def __init__(self, task):
    super().__init__(task)
    self.action_space = gym.spaces.Box(-1.0, 1.0, task.action_space.shape, np.float32)

  def step(self, action):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != self.action_space.shape:
      raise ValueError(f"action shape {action.shape} != {self.action_space.shape}")
    observation, reward, terminated, truncated, info = super().step(
      (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
    )
    info["policy_action"] = action
    return observation, reward, terminated, truncated, info
