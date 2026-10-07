"""Train and accept SAC reacher used by high-level options."""

from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("SAC/reacher.py", {})
DEFAULTS = SETTINGS["defaults"].get("SAC/reacher.py", {})

import argparse
import csv
import hashlib
import json
import platform
from numbers import Integral, Real
from pathlib import Path

import gymnasium as gym
import motornet as mn
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .core.rich_progress import RichProgressCallback


DT = PARAMS["DT"]
OBSERVATION_SIZE = PARAMS["OBSERVATION_SIZE"]
ACTION_SIZE = PARAMS["ACTION_SIZE"]
WORKSPACE_RADIUS = PARAMS["WORKSPACE_RADIUS"]
MIN_SEPARATION = PARAMS["MIN_SEPARATION"]
HIT_RADIUS = PARAMS["HIT_RADIUS"]
MAX_EPISODE_STEPS = PARAMS["MAX_EPISODE_STEPS"]
DEFAULT_RUN_PATH = Path(__file__).resolve().parents[1] / "results/option_reacher"
DEFAULT_SEED = PARAMS["DEFAULT_SEED"]
DEFAULT_EVAL_SEED = PARAMS["DEFAULT_EVAL_SEED"]
DEFAULT_TIMESTEPS = PARAMS["DEFAULT_TIMESTEPS"]
DEFAULT_EVAL_FREQUENCY = PARAMS["DEFAULT_EVAL_FREQUENCY"]
DEFAULT_EVAL_COUNT = PARAMS["DEFAULT_EVAL_COUNT"]


def sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as source:
    for block in iter(lambda: source.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def action_to_excitation(action):
  action = np.asarray(action, dtype=np.float32)
  if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all():
    raise ValueError("action must be a finite six-dimensional vector")
  return (np.clip(action, -1., 1.) + 1.) / 2.


def sample_disk(rng, center, radius=WORKSPACE_RADIUS, size=None):
  center = np.asarray(center, dtype=np.float64)
  if center.shape != (2,) or not np.isfinite(center).all():
    raise ValueError("center must contain two finite values")
  if not isinstance(radius, Real) or isinstance(radius, bool) or not np.isfinite(radius) or radius <= 0:
    raise ValueError("radius must be positive and finite")
  if size is not None and (not isinstance(size, Integral) or isinstance(size, bool) or size < 1):
    raise ValueError("size must be a positive integer")
  rng = np.random.default_rng(rng) if isinstance(rng, Integral) else rng
  shape = None if size is None else int(size)
  distance = float(radius) * np.sqrt(rng.random(shape))
  angle = 2 * np.pi * rng.random(shape)
  return (center + np.stack((distance * np.cos(angle), distance * np.sin(angle)), axis=-1)).astype(np.float32)


def sample_start_target(rng, center, radius=WORKSPACE_RADIUS, min_separation=MIN_SEPARATION):
  rng = np.random.default_rng(rng) if isinstance(rng, Integral) else rng
  start = sample_disk(rng, center, radius)
  for _ in range(10000):
    target = sample_disk(rng, center, radius)
    if np.linalg.norm(start - target) >= min_separation:
      return start, target
  raise RuntimeError("unable to sample separated start and target")


def sample_pairs(seed, center, count=DEFAULTS["sample_pairs"]["count"], radius=WORKSPACE_RADIUS,
                 min_separation=MIN_SEPARATION):
  if not isinstance(count, Integral) or isinstance(count, bool) or count < 1:
    raise ValueError("count must be a positive integer")
  rng = np.random.default_rng(seed)
  return np.asarray([
    sample_start_target(rng, center, radius, min_separation)
    for _ in range(int(count))
  ], dtype=np.float32)


def _inverse_kinematics(xy, l1, l2, lower, upper):
  xy = np.asarray(xy, dtype=np.float64)
  if xy.shape != (2,) or not np.isfinite(xy).all():
    raise ValueError("xy must contain two finite values")
  cosine = (xy @ xy - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
  if abs(cosine) > 1 + 1e-6:
    raise ValueError("XY coordinate outside Arm26 workspace")
  elbow = np.arccos(np.clip(cosine, -1., 1.))
  shoulder = np.arctan2(xy[1], xy[0]) - np.arctan2(
    l2 * np.sin(elbow), l1 + l2 * np.cos(elbow)
  )
  joints = np.asarray((shoulder, elbow), dtype=np.float32)
  if np.any(joints < np.asarray(lower)[:2] - 1e-6) or np.any(joints > np.asarray(upper)[:2] + 1e-6):
    raise ValueError("XY coordinate violates Arm26 joint limits")
  return joints


class ReacherTask(mn.environment.Environment):
  """Single-target Arm26 task with 22-D observation and SAC reward."""

  def __init__(self, effector, *, workspace_radius=WORKSPACE_RADIUS,
               min_separation=MIN_SEPARATION, hit_radius=HIT_RADIUS,
               max_episode_steps=MAX_EPISODE_STEPS):
    if type(effector) is not mn.effector.RigidTendonArm26:
      raise TypeError("effector must be a RigidTendonArm26")
    if not np.isclose(effector.dt, DT):
      raise ValueError("effector timestep must be 0.01")
    if any(not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value) or value <= 0
           for value in (workspace_radius, min_separation, hit_radius)):
      raise ValueError("workspace radius, separation, and hit radius must be positive and finite")
    if not isinstance(max_episode_steps, Integral) or isinstance(max_episode_steps, bool) or max_episode_steps < 1:
      raise ValueError("max_episode_steps must be a positive integer")
    super().__init__(
      effector=effector, name="ReacherTask", differentiable=False,
      max_ep_duration=int(max_episode_steps) * effector.dt,
      action_noise=0., obs_noise=0., proprioception_delay=0., vision_delay=0.,
    )
    self.workspace_radius = float(workspace_radius)
    self.min_separation = float(min_separation)
    self.hit_radius = float(hit_radius)
    self.max_episode_steps = int(max_episode_steps)
    reference = torch.tensor([[*self.reference_posture, 0., 0.]], dtype=torch.float32)
    self.register_buffer("workspace_center", self.joint2cartesian(reference)[0, :2])
    self.subgoal = torch.zeros((1, 2), dtype=torch.float32)
    self.start_xy = torch.zeros((1, 2), dtype=torch.float32)
    self.finished = torch.zeros(1, dtype=torch.bool)
    self.elapsed_steps = 0

  reference_posture = np.deg2rad([45., 90.]).astype(np.float32)

  def _get_obs_size(self):
    return OBSERVATION_SIZE

  def inverse_kinematics(self, xy):
    joints = _inverse_kinematics(
      xy, self.skeleton.L1, self.skeleton.L2,
      self.effector.pos_lower_bound.detach().cpu().numpy(),
      self.effector.pos_upper_bound.detach().cpu().numpy(),
    )
    state = torch.tensor([[*joints, 0., 0.]], dtype=torch.float32)
    if not np.allclose(self.joint2cartesian(state)[0, :2].detach().cpu().numpy(), xy, atol=1e-6):
      raise ValueError("inverse-kinematics reconstruction failed")
    return joints

  def get_obs(self, action=None, deterministic=DEFAULTS["ReacherTask.get_obs"]["deterministic"]):
    del action, deterministic
    fingertip = self.states["fingertip"]
    muscle = self.states["muscle"]
    lengths = (muscle[:, 1:2, :] / self.muscle.l0_ce).squeeze(1)
    velocities = (muscle[:, 2:3, :] / self.muscle.vmax).squeeze(1)
    activation = muscle[:, 0, :]
    observation = torch.cat((self.subgoal, fingertip, lengths, velocities, activation), dim=-1)
    return self.detach(observation)

  def _info(self, **values):
    info = {
      "states": self._maybe_detach_states(),
      "subgoal_xy": self.detach(self.subgoal),
      "start_xy": self.detach(self.start_xy),
      "finished": self.detach(self.finished),
      "is_success": self.detach(self.finished),
    }
    info.update({key: self.detach(value) if torch.is_tensor(value) else value
                 for key, value in values.items()})
    return info

  def reset(self, *, seed=None, options=None):
    self._set_generator(seed=seed)
    options = {} if options is None else dict(options)
    center = self.workspace_center.detach().cpu().numpy()
    if "start_xy" in options or "target_xy" in options:
      if "start_xy" not in options or "target_xy" not in options:
        raise ValueError("start_xy and target_xy must be supplied together")
      start = np.asarray(options["start_xy"], dtype=np.float32)
      target = np.asarray(options["target_xy"], dtype=np.float32)
      if start.shape != (2,) or target.shape != (2,) or not np.isfinite(start).all() or not np.isfinite(target).all():
        raise ValueError("start_xy and target_xy must contain two finite values")
      if any(np.linalg.norm(point - center) > self.workspace_radius + 1e-6 for point in (start, target)):
        raise ValueError("start_xy and target_xy must lie inside workspace disk")
      if np.linalg.norm(start - target) < self.min_separation:
        raise ValueError("start and target must be separated")
    else:
      start, target = sample_start_target(
        self.np_random, center, self.workspace_radius, self.min_separation,
      )
    start_joints = self.inverse_kinematics(start)
    joint_state = torch.tensor([[*start_joints, 0., 0.]], dtype=torch.float32)
    self.subgoal = torch.tensor(target[None], dtype=torch.float32)
    self.start_xy = torch.tensor(start[None], dtype=torch.float32)
    self.finished = torch.zeros(1, dtype=torch.bool)
    self.elapsed_steps = 0
    super().reset(seed=None, options={
      "batch_size": 1, "joint_state": joint_state, "deterministic": True,
    })
    self.goal = self.subgoal
    observation = self.get_obs(deterministic=True)
    info = self._info(distance_before=np.linalg.norm(start - target))
    return observation, info

  def step(self, action, deterministic=DEFAULTS["ReacherTask.step"]["deterministic"], **kwargs):
    del deterministic
    action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
    if action.ndim == 1:
      action = action[None, :]
    if action.shape != (1, ACTION_SIZE) or not torch.isfinite(action).all() or torch.any(action < 0) or torch.any(action > 1):
      raise ValueError("task action must be one finite six-dimensional excitation vector in [0, 1]")
    self.elapsed_steps += 1
    self.elapsed = self.elapsed_steps * self.dt
    before = self.states["fingertip"].clone()
    before_distance = torch.linalg.vector_norm(before - self.subgoal, dim=1)
    self.effector.step(action, **kwargs)
    after = self.states["fingertip"].clone()
    after_distance = torch.linalg.vector_norm(after - self.subgoal, dim=1)
    progress = (before_distance - after_distance) / 0.01
    step_distance = torch.linalg.vector_norm(after - before, dim=1)
    wasted = torch.clamp(step_distance - (before_distance - after_distance), min=0.)
    distance_reward = -0.1 * after_distance / 0.10
    path_reward = -0.05 * wasted / 0.01
    hit = (after_distance <= self.hit_radius) & ~self.finished
    self.finished |= hit
    target_reward = 20. * hit.to(torch.float32)
    completion_reward = 200. * hit.to(torch.float32)
    reward = progress + distance_reward + path_reward + target_reward + completion_reward
    observation = self.get_obs(deterministic=True)
    terminated = bool(torch.all(self.finished).item())
    truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
    info = self._info(
      fingertip=after, distance_before=before_distance, distance_after=after_distance,
      progress_reward=progress, distance_reward=distance_reward,
      path_reward=path_reward, wasted_distance=wasted, step_distance=step_distance,
      target_reward=target_reward, completion_reward=completion_reward,
      target_hit=hit, timeout=truncated,
    )
    return observation, self.detach(reward[:, None]), terminated, truncated, info


def _squeeze(value):
  if isinstance(value, dict):
    return {key: _squeeze(item) for key, item in value.items()}
  if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 1:
    return value[0]
  return value


class ReacherGym(gym.Env):
  metadata = {"render_modes": []}

  def __init__(self, task):
    self.task = task
    self.observation_space = task.observation_space
    self.action_space = gym.spaces.Box(-1., 1., (ACTION_SIZE,), dtype=np.float32)

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    observation, info = self.task.reset(seed=seed, options=options)
    info = _squeeze(info)
    info["is_success"] = bool(info["finished"])
    return np.asarray(observation[0], dtype=np.float32), info

  def step(self, action):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all():
      raise ValueError("action must be a finite six-dimensional vector")
    observation, reward, terminated, truncated, info = self.task.step(
      action_to_excitation(action)[None, :], deterministic=True,
    )
    info = _squeeze(info)
    info["policy_action"] = action
    info["excitation"] = action_to_excitation(action)
    info["is_success"] = bool(info["finished"])
    return np.asarray(observation[0], dtype=np.float32), float(reward[0, 0]), bool(terminated), bool(truncated), info

  def close(self):
    return None


def make_env():
  effector = mn.effector.RigidTendonArm26(
    muscle=mn.muscle.RigidTendonHillMuscle(), timestep=DT,
  )
  return ReacherGym(ReacherTask(effector))


ReacherEnv = ReacherTask
SACReacherEnv = ReacherTask
SACReacherGym = ReacherGym
ReachEnv = ReacherTask
ReachGym = ReacherGym


def _line_deviation(point, start, target):
  direction = target - start
  denominator = float(direction @ direction)
  projection = 0. if denominator == 0 else np.clip((point - start) @ direction / denominator, 0., 1.)
  return float(np.linalg.norm(point - (start + projection * direction)))


def _attempt(model, env, pair, index=0, failure_dir=None):
  start, target = np.asarray(pair[0]), np.asarray(pair[1])
  observation, info = env.reset(options={"start_xy": start, "target_xy": target})
  path = 0.
  deviation = 0.
  observations, actions, rewards, fingertips = [observation.copy()], [], [], []
  for step in range(env.task.max_episode_steps):
    action, _ = model.predict(observation, deterministic=True)
    next_observation, reward, terminated, truncated, next_info = env.step(action)
    point = np.asarray(next_info["states"]["fingertip"], dtype=np.float32)
    path += float(next_info["step_distance"])
    deviation = max(deviation, _line_deviation(point, start, target))
    observations.append(next_observation.copy())
    actions.append(np.asarray(action, dtype=np.float32).copy())
    rewards.append(float(reward))
    fingertips.append(point.copy())
    observation, info = next_observation, next_info
    if terminated or truncated:
      break
  direct = max(float(np.linalg.norm(target - start)), 1e-8)
  result = {
    "index": int(index), "start_xy": start.tolist(), "target_xy": target.tolist(),
    "success": bool(info["is_success"]), "duration": step + 1,
    "return": float(np.sum(rewards)), "path_ratio": path / direct,
    "max_deviation": float(deviation), "final_distance": float(info["distance_after"]),
    "terminated": bool(terminated), "truncated": bool(truncated),
  }
  if failure_dir is not None and not result["success"]:
    failure_dir = Path(failure_dir)
    failure_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      failure_dir / f"episode_{index:04d}.npz",
      observation=np.asarray(observations), action=np.asarray(actions),
      reward=np.asarray(rewards), fingertip=np.asarray(fingertips),
      start_xy=start, target_xy=target,
    )
  return result


def summarize_attempts(attempts):
  if not attempts:
    raise ValueError("attempts must not be empty")
  successful = [item for item in attempts if item["success"]]
  return {
    "attempts": attempts,
    "success_rate": float(np.mean([item["success"] for item in attempts])),
    "mean_path_ratio": float(np.mean([item["path_ratio"] for item in successful])) if successful else float("inf"),
    "max_deviation": float(max(item["max_deviation"] for item in attempts)),
    "mean_successful_duration": float(np.mean([item["duration"] for item in successful])) if successful else float("inf"),
    "accepted": bool(np.mean([item["success"] for item in attempts]) >= .99),
  }


def evaluate_pairs(model, env, pairs, output_dir, *, failure_dir=None):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  attempts = [_attempt(model, env, pair, index, failure_dir) for index, pair in enumerate(pairs)]
  summary = summarize_attempts(attempts)
  (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  return summary


def checkpoint_key(summary):
  return (
    float(summary["success_rate"]), -float(summary["mean_path_ratio"]),
    -float(summary["max_deviation"]), -float(summary["mean_successful_duration"]),
  )


class ReacherEvalCallback(BaseCallback):
  def __init__(self, eval_freq, pairs, output_dir, display=None):
    super().__init__()
    self.eval_freq = int(eval_freq)
    self.pairs = np.asarray(pairs, dtype=np.float32)
    self.output_dir = Path(output_dir)
    self.history = []
    self.best = None
    self.best_success = -1.
    self.quality_streak = 0
    self.stopped_early = False
    self.display = display

  def _init_callback(self):
    self.eval_env = make_env()

  def _evaluate(self, name):
    if self.display:
      self.display.set_status("evaluating")
    summary = evaluate_pairs(
      self.model, self.eval_env, self.pairs, self.output_dir / name,
      failure_dir=self.output_dir / name / "failures",
    )
    record = {key: value for key, value in summary.items() if key != "attempts"}
    record["step"] = int(self.num_timesteps)
    self.history.append(record)
    quality = (
      summary["success_rate"] >= .99
      and summary["mean_path_ratio"] <= 1.2
      and summary["max_deviation"] <= .02
    )
    self.quality_streak = self.quality_streak + 1 if quality else 0
    if self.best is None or checkpoint_key(summary) > checkpoint_key(self.best):
      self.best = summary
      self.best_success = summary["success_rate"]
      self.model.save(str(self.output_dir.parent / "best_model"))
    if self.display:
      self.display.set_eval(
        summary["success_rate"], self.best_success,
        len(self.history), self.quality_streak,
      )
      self.display.set_status("training")
    (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")

  def _on_step(self):
    if self.n_calls % self.eval_freq == 0:
      self._evaluate(f"step_{self.num_timesteps:08d}")
      if self.quality_streak >= 3:
        self.stopped_early = True
        if self.display:
          self.display.set_status("early stop")
        return False
    return True

  def _on_training_end(self):
    self._evaluate("final")
    self.eval_env.close()
    if self.display:
      self.display.set_status("complete")
      self.display.close()


def _training_env(run_path):
  def factory():
    return Monitor(make_env(), filename=str(run_path / "monitor.csv"), info_keywords=("is_success",))
  return factory


def train(run_path=DEFAULT_RUN_PATH, timesteps=DEFAULT_TIMESTEPS, seed=DEFAULT_SEED,
          evaluation_frequency=DEFAULT_EVAL_FREQUENCY, evaluation_count=DEFAULT_EVAL_COUNT):
  run_path = Path(run_path)
  if run_path.exists():
    raise FileExistsError(run_path)
  run_path.mkdir(parents=True)
  center_env = make_env()
  center = center_env.task.workspace_center.detach().cpu().numpy()
  pairs = sample_pairs(DEFAULT_EVAL_SEED, center, evaluation_count)
  train_env = DummyVecEnv([_training_env(run_path)])
  display = RichProgressCallback(run_path / "loss_history.csv", algo="sac")
  callback = ReacherEvalCallback(
    evaluation_frequency, pairs, run_path / "evaluations", display,
  )
  model = SAC(
    "MlpPolicy", train_env, **SETTINGS["training"]["sac"], seed=seed, verbose=0,
  )
  model.learn(total_timesteps=int(timesteps), callback=[display, callback])
  model.save(str(run_path / "final_model"))
  config = {
    "algorithm": "sac", "seed": int(seed), "requested_timesteps": int(timesteps),
    "actual_timesteps": int(model.num_timesteps), "observation_order": [
      "subgoal_xy", "fingertip_xy", "normalized_lengths", "normalized_velocities", "activations",
    ], "observation_shape": [OBSERVATION_SIZE], "action_shape": [ACTION_SIZE],
    "action_range": [-1., 1.], "excitation_mapping": "(action + 1) / 2",
    "environment": {"workspace_radius": WORKSPACE_RADIUS, "min_separation": MIN_SEPARATION,
                     "hit_radius": HIT_RADIUS, "max_episode_steps": MAX_EPISODE_STEPS,
                     "action_noise": 0., "observation_noise": 0.},
    "reward": {"progress_scale": .01, "distance_scale": .1, "path_scale": .05,
               "target_bonus": 20., "completion_bonus": 200.},
    "evaluation": {"frequency": int(evaluation_frequency), "count": int(evaluation_count),
                    "seed": DEFAULT_EVAL_SEED, "success_threshold": .99,
                    "path_ratio_threshold": 1.2, "deviation_threshold": .02,
                    "streak": 3, "stopped_early": callback.stopped_early},
    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                 "torch": str(torch.__version__)},
  }
  config["checkpoint_hashes"] = {
    "best_model.zip": sha256(run_path / "best_model.zip"),
    "final_model.zip": sha256(run_path / "final_model.zip"),
  }
  (run_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
  train_env.close(); center_env.close()
  return run_path / "final_model.zip"


def build_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)
  train_parser = subparsers.add_parser("train")
  train_parser.add_argument("--run-path", type=Path, required=True)
  train_parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
  train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
  train_parser.add_argument("--evaluation-frequency", type=int, default=DEFAULT_EVAL_FREQUENCY)
  train_parser.add_argument("--evaluation-count", type=int, default=DEFAULT_EVAL_COUNT)
  return parser


def main():
  args = build_parser().parse_args()
  if args.command == "train":
    print(train(args.run_path, args.timesteps, args.seed,
                 args.evaluation_frequency, args.evaluation_count))


if __name__ == "__main__":
  main()
