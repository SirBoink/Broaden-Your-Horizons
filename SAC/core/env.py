
from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("SAC/core/env.py", {})
DEFAULTS = SETTINGS["defaults"].get("SAC/core/env.py", {})
from numbers import Integral
from typing import Any, Sequence

import motornet as mn
import numpy as np
import torch

sequence = tuple(PARAMS["sequence"])
radius = PARAMS["radius"]
target_hit_radius = PARAMS["target_hit_radius"]


class SequentialReachEnv(mn.environment.Environment):
  """Sequential reaches through fixed Arm26 targets."""

  target_names = tuple(f"T{i}" for i in range(8))
  reference_posture = np.deg2rad([45.0, 90.0]).astype(np.float32)

  def __init__(
    self,
    effector: mn.effector.RigidTendonArm26,
    sequence: Sequence[str | int],
    *,
    hit_radius: float = target_hit_radius,
    angular_offset_deg: float = DEFAULTS["SequentialReachEnv.__init__"]["angular_offset_deg"],
    dwell: bool = DEFAULTS["SequentialReachEnv.__init__"]["dwell"],
    tolerance: float = DEFAULTS["SequentialReachEnv.__init__"]["tolerance"],
    hold_steps: int = DEFAULTS["SequentialReachEnv.__init__"]["hold_steps"],
    max_episode_steps: int = DEFAULTS["SequentialReachEnv.__init__"]["max_episode_steps"],
    progress_reward_scale: float = DEFAULTS["SequentialReachEnv.__init__"]["progress_reward_scale"],
    straight_path_reward: bool = DEFAULTS["SequentialReachEnv.__init__"]["straight_path_reward"],
    action_noise: float = DEFAULTS["SequentialReachEnv.__init__"]["action_noise"],
    obs_noise: float = DEFAULTS["SequentialReachEnv.__init__"]["obs_noise"],
    proprioception_delay: float | None = None,
    vision_delay: float | None = None,
    **kwargs,
  ):
    if not isinstance(effector, mn.effector.RigidTendonArm26):
      raise TypeError("effector must be a RigidTendonArm26")
    if (not np.isfinite(hit_radius) or hit_radius <= 0 or tolerance <= 0
        or hold_steps < 1 or progress_reward_scale < 0
        or not isinstance(max_episode_steps, Integral)
        or isinstance(max_episode_steps, bool) or max_episode_steps < 1):
      raise ValueError(
        "hit_radius, tolerance, and hold_steps must be positive; "
        "progress_reward_scale must be non-negative; "
        "max_episode_steps must be a positive integer"
      )

    self.default_sequence = self._parse_sequence(sequence)
    self.dwell = dwell
    self.hit_radius = float(hit_radius)
    self.tolerance = tolerance
    self.hold_steps = hold_steps
    self.max_episode_steps = int(max_episode_steps)
    self.progress_reward_scale = float(progress_reward_scale)
    self.straight_path_reward = bool(straight_path_reward)
    super().__init__(
      effector=effector,
      name="SequentialReachEnv",
      max_ep_duration=max_episode_steps * effector.dt,
      action_noise=action_noise,
      obs_noise=obs_noise,
      proprioception_delay=proprioception_delay,
      vision_delay=vision_delay,
      **kwargs,
    )

    joint_state = torch.tensor([[*self.reference_posture, 0.0, 0.0]], device=self.device)
    center = self.joint2cartesian(joint_state)[:, :2].squeeze(0)
    angles = torch.deg2rad(
      torch.arange(8, dtype=torch.float32, device=self.device) * 45 + angular_offset_deg
    )
    targets = center + radius * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    self.register_buffer("target_center", center)
    self.register_buffer("target_angles", angles)
    self.register_buffer("targets", targets)
    self.register_buffer("target_joint_angles", self._reachable_joint_angles(targets))

  def _get_obs_size(self) -> int:
    return 2 + len(self.target_names) + 2 + 2 * self.effector.n_muscles + (
      self.effector.n_muscles * self.action_frame_stacking
    )

  def get_obs(self, action=None, deterministic: bool = DEFAULTS["SequentialReachEnv.get_obs"]["deterministic"]):
    self.update_obs_buffer(action=action)
    physical = torch.cat([
      self.goal,
      self.obs_buffer["vision"][0],
      self.obs_buffer["proprioception"][0],
    ] + self.obs_buffer["action"][:self.action_frame_stacking], dim=-1)
    if not deterministic:
      one_hot_size = len(self.target_names)
      noise = self.obs_noise[:2] + self.obs_noise[2 + one_hot_size:]
      physical = self.apply_noise(physical, noise=noise)
    target_one_hot = torch.nn.functional.one_hot(
      self.sequence_index, num_classes=len(self.target_names)
    ).to(torch.float32)
    obs = torch.cat([physical[:, :2], target_one_hot, physical[:, 2:]], dim=-1)
    return obs if self.differentiable else self.detach(obs)

  @classmethod
  def _parse_sequence(cls, sequence: Sequence[str | int]) -> tuple[int, ...]:
    parsed = []
    for target in sequence:
      if isinstance(target, str) and target in cls.target_names:
        parsed.append(int(target[1:]))
      elif isinstance(target, Integral) and not isinstance(target, bool) and 0 <= target < 8:
        parsed.append(int(target))
      else:
        raise ValueError(f"Unknown target {target!r}; use T0-T7 or indices 0-7")
    if not parsed:
      raise ValueError("sequence must contain at least one target")
    return tuple(parsed)

  def _reachable_joint_angles(self, targets: torch.Tensor) -> torch.Tensor:
    xy = targets.detach().cpu().numpy()
    l1, l2 = self.skeleton.L1, self.skeleton.L2
    cos_elbow = (np.sum(xy**2, axis=1) - l1**2 - l2**2) / (2 * l1 * l2)
    if np.any(np.abs(cos_elbow) > 1 + 1e-6):
      raise ValueError("At least one target lies outside the Arm26 workspace")

    elbow = np.arccos(np.clip(cos_elbow, -1, 1))
    shoulder = np.arctan2(xy[:, 1], xy[:, 0]) - np.arctan2(
      l2 * np.sin(elbow), l1 + l2 * np.cos(elbow)
    )
    joints = np.column_stack((shoulder, elbow)).astype(np.float32)
    lower = self.effector.pos_lower_bound.detach().cpu().numpy()
    upper = self.effector.pos_upper_bound.detach().cpu().numpy()
    if np.any(joints < lower) or np.any(joints > upper):
      raise ValueError("At least one target violates the Arm26 joint limits")

    joint_state = torch.tensor(
      np.column_stack((joints, np.zeros_like(joints))), dtype=torch.float32, device=self.device
    )
    reconstructed = self.joint2cartesian(joint_state)[:, :2]
    if not torch.allclose(reconstructed, targets, atol=1e-6):
      raise ValueError("Arm26 inverse-kinematics reachability check failed")
    return torch.tensor(joints, dtype=torch.float32, device=self.device)

  def reset(
    self, *, seed: int | None = None, options: dict[str, Any] | None = None
  ) -> tuple[Any, dict[str, Any]]:
    options = {} if options is None else options
    if "joint_state" in options:
      raise ValueError("SequentialReachEnv always starts at the reference posture")

    batch_size = int(options.get("batch_size", 1))
    if batch_size < 1:
      raise ValueError("batch_size must be positive")
    self.sequence = self._parse_sequence(options.get("sequence", self.default_sequence))
    self.elapsed_steps = 0
    self._sequence = torch.tensor(self.sequence, dtype=torch.long, device=self.device)
    self.sequence_index = torch.zeros(batch_size, dtype=torch.long, device=self.device)
    self.hold_count = torch.zeros(batch_size, dtype=torch.long, device=self.device)
    self.finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    joint_state = torch.tensor(self.reference_posture, device=self.device).repeat(batch_size, 1)
    base_options = {
      "batch_size": batch_size,
      "joint_state": joint_state,
      "deterministic": options.get("deterministic", False),
    }
    _, info = super().reset(seed=seed, options=base_options)
    self.goal = self.targets[self._sequence[0]].repeat(batch_size, 1)
    obs = self.get_obs(deterministic=base_options["deterministic"])
    info.update(self._task_info())
    return obs, info

  def step(
    self,
    action: torch.Tensor | np.ndarray,
    deterministic: bool = DEFAULTS["SequentialReachEnv.step"]["deterministic"],
    **kwargs,
  ) -> tuple[torch.Tensor | np.ndarray, np.ndarray | None, bool, bool, dict[str, Any]]:
    self.elapsed_steps += 1
    self.elapsed = self.elapsed_steps * self.dt
    action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
    noisy_action = action if deterministic else self.apply_noise(action, self.action_noise)
    evaluated_goal = self.goal.clone()
    evaluated_target = self._sequence[self.sequence_index].clone()
    active = ~self.finished
    before_hand = self.states["fingertip"].clone()
    before_error = self.states["fingertip"] - evaluated_goal
    before_distance = torch.linalg.vector_norm(before_error, dim=1)
    self.effector.step(noisy_action, **kwargs)

    after_hand = self.states["fingertip"]
    evaluated_error = self.states["fingertip"] - evaluated_goal
    after_distance = torch.linalg.vector_norm(evaluated_error, dim=1)
    progress_dist = before_distance - after_distance
    progress_reward = self.progress_reward_scale * progress_dist / self.tolerance
    progress_reward = torch.where(active, progress_reward, torch.zeros_like(progress_reward))
    step_dist = torch.linalg.vector_norm(after_hand - before_hand, dim=1)
    wasted_dist = torch.clamp(step_dist - progress_dist, min=0.0)
    lambda_path = 0.5 * self.progress_reward_scale if self.straight_path_reward else 0.0
    path_penalty = lambda_path * (wasted_dist / self.tolerance)
    path_penalty = torch.where(active, path_penalty, torch.zeros_like(path_penalty))
    travel = step_dist / self.dt
    travel_reward = -path_penalty
    within = (torch.linalg.vector_norm(evaluated_error, dim=1) <= self.hit_radius) & ~self.finished
    if self.dwell:
      self.hold_count = torch.where(within, self.hold_count + 1, 0)
      completed = (self.hold_count >= self.hold_steps) & ~self.finished
    else:
      completed = within

    last = self.sequence_index == len(self.sequence) - 1
    self.finished |= completed & last
    self.sequence_index += (completed & ~last).long()
    self.hold_count = torch.where(completed, 0, self.hold_count)
    self.goal = self.targets[self._sequence[self.sequence_index]]

    obs = self.get_obs(action=noisy_action, deterministic=deterministic)
    target_reward = 20.0 * completed.float()
    completion_reward = 200.0 * (completed & last).float()
    reward_tensor = progress_reward - path_penalty + target_reward + completion_reward
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
      "evaluated_target_error": (
        evaluated_error if self.differentiable else self.detach(evaluated_error)
      ),
      "distance_before": before_distance if self.differentiable else self.detach(before_distance),
      "distance_after": after_distance if self.differentiable else self.detach(after_distance),
      "progress_dist": progress_dist if self.differentiable else self.detach(progress_dist),
      "step_dist": step_dist if self.differentiable else self.detach(step_dist),
      "wasted_dist": wasted_dist if self.differentiable else self.detach(wasted_dist),
      "progress_reward": progress_reward if self.differentiable else self.detach(progress_reward),
      "path_penalty": path_penalty if self.differentiable else self.detach(path_penalty),
      "travel": travel if self.differentiable else self.detach(travel),
      "travel_reward": travel_reward if self.differentiable else self.detach(travel_reward),
      "target_reward": target_reward if self.differentiable else self.detach(target_reward),
      "completion_reward": completion_reward if self.differentiable else self.detach(completion_reward),
      "completed": completed if self.differentiable else self.detach(completed),
      "timeout": timeout,
      **self._task_info(),
    }
    return obs, reward, terminated, truncated, info

  def _task_info(self) -> dict[str, Any]:
    activation = self.states["muscle"][:, 0, :]
    target_error = self.states["fingertip"] - self.goal
    values = {
      "goal": self.goal,
      "sequence_index": self.sequence_index,
      "target": self._sequence[self.sequence_index],
      "hold_count": self.hold_count,
      "finished": self.finished,
      "target_error": target_error,
      "target_distance": torch.linalg.vector_norm(target_error, dim=1),
      "activation": activation,
      "activation_effort": torch.mean(activation.square(), dim=1),
    }
    return values if self.differentiable else {key: self.detach(value) for key, value in values.items()}
