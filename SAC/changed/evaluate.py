
from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("SAC/changed/evaluate.py", {})
DEFAULTS = SETTINGS["defaults"].get("SAC/changed/evaluate.py", {})
import argparse
import json
from pathlib import Path

import motornet as mn
import numpy as np
from stable_baselines3 import PPO, SAC

from SAC.core.env import target_hit_radius

from .env import StraightReachEnv, StraightReachGym, sequence as default_sequence


def make_env(sequence=default_sequence, hit_radius: float = target_hit_radius,
             random_start: bool = DEFAULTS["make_env"]["random_start"], *, start_pos=None, path_penalty_scale=DEFAULTS["make_env"]["path_penalty_scale"]):
  effector = mn.effector.RigidTendonArm26(
    muscle=mn.muscle.RigidTendonHillMuscle(), timestep=0.01
  )
  return StraightReachGym(StraightReachEnv(
    effector, sequence, hit_radius=hit_radius, differentiable=False,
    action_noise=0.0, obs_noise=0.0, random_start=random_start,
    start_pos=start_pos, path_penalty_scale=path_penalty_scale,
  ))


def _line_deviation(point, start, goal):
  direction = goal - start
  scale = float(direction @ direction)
  projection = 0.0 if scale == 0 else np.clip((point - start) @ direction / scale, 0, 1)
  return float(np.linalg.norm(point - (start + projection * direction)))


def run_evaluation(model, env, output_dir: Path, episodes: int = DEFAULTS["run_evaluation"]["episodes"],
                   save_episodes: bool = DEFAULTS["run_evaluation"]["save_episodes"]):
  output_dir.mkdir(parents=True, exist_ok=True)
  summaries = []
  transition_keys = (
    "distance_before", "distance_after", "progress_dist", "step_dist", "wasted_dist",
    "progress_reward", "distance_reward", "path_penalty", "travel", "travel_reward",
    "target_reward", "completion_reward", "reward", "completed", "terminated", "truncated",
  )
  for episode in range(episodes):
    obs, info = env.reset(seed=episode)
    state_keys = ("fingertip", "joint", "muscle")
    states = {key: [np.asarray(info["states"][key], dtype=np.float32)] for key in state_keys}
    states.update({
      "activation": [np.asarray(info["activation"], dtype=np.float32)],
      "goal": [np.asarray(info["goal"], dtype=np.float32)],
      "target_index": [int(info["sequence_index"])],
      "target_id": [int(info["target"])],
    })
    transitions = {key: [] for key in transition_keys}
    transitions.update({"action": [], "applied_action": [], "evaluated_goal": [],
                        "evaluated_target": []})
    leg_start = states["fingertip"][0]
    leg_goal = states["goal"][0]
    direct_distance = float(np.linalg.norm(leg_goal - leg_start))
    path_length = 0.0
    max_deviation = 0.0
    path_ratios, deviations = [], []

    for step in range(env.task.max_episode_steps):
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, info = env.step(action)
      transitions["action"].append(np.asarray(action, dtype=np.float32))
      transitions["applied_action"].append(np.asarray(info["noisy action"], dtype=np.float32))
      transitions["evaluated_goal"].append(np.asarray(info["evaluated_goal"], dtype=np.float32))
      transitions["evaluated_target"].append(int(info["evaluated_target"]))
      for key in transition_keys:
        value = {"reward": reward, "terminated": terminated, "truncated": truncated}.get(key, info.get(key))
        transitions[key].append(value)
      for key in state_keys:
        states[key].append(np.asarray(info["states"][key], dtype=np.float32))
      states["activation"].append(np.asarray(info["activation"], dtype=np.float32))
      states["goal"].append(np.asarray(info["goal"], dtype=np.float32))
      states["target_index"].append(int(info["sequence_index"]))
      states["target_id"].append(int(info["target"]))

      point = states["fingertip"][-1]
      path_length += float(info["step_dist"])
      max_deviation = max(max_deviation, _line_deviation(point, leg_start, leg_goal))
      if bool(info["completed"]):
        path_ratios.append(path_length / max(direct_distance, 1e-8))
        deviations.append(max_deviation)
        leg_start = point
        leg_goal = states["goal"][-1]
        direct_distance = float(np.linalg.norm(leg_goal - leg_start))
        path_length = max_deviation = 0.0
      if terminated or truncated:
        break

    if save_episodes:
      np.savez_compressed(
        output_dir / f"episode_{episode:03d}.npz",
        time=np.arange(step + 2, dtype=np.float32) * env.task.dt,
        **{key: np.stack(value) for key, value in states.items()},
        **{key: np.asarray(value) for key, value in transitions.items()},
      )
    summaries.append({
      "episode": episode,
      "steps": step + 1,
      "success": bool(info["finished"]),
      "return": float(np.sum(transitions["reward"])),
      "completed_targets": len(path_ratios),
      "mean_path_ratio": float(np.mean(path_ratios)) if path_ratios else None,
      "max_deviation": float(max(deviations)) if deviations else None,
    })
  (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
  return summaries


def evaluate(model_path: Path, output_dir: Path, episodes: int = DEFAULTS["evaluate"]["episodes"], *,
             algo: str = DEFAULTS["evaluate"]["algo"], sequence=default_sequence,
             hit_radius: float = target_hit_radius, random_start: bool = DEFAULTS["evaluate"]["random_start"],
             start_pos=None):
  model_class = SAC if algo == "sac" else PPO
  model = model_class.load(str(model_path))
  env = make_env(sequence, hit_radius, random_start, start_pos=start_pos)
  summaries = run_evaluation(model, env, output_dir, episodes)
  env.close()
  return summaries


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("model", type=Path)
  parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
  parser.add_argument("--output-dir", type=Path, default=Path("runs/changed_eval"))
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--sequence", type=int, nargs="+", default=list(default_sequence))
  parser.add_argument("--hit-radius", type=float, default=target_hit_radius)
  parser.add_argument("--fixed-start", action="store_true")
  parser.add_argument("--start-pos", type=float, nargs=2,
                      metavar=("SHOULDER_DEG", "ELBOW_DEG"))
  args = parser.parse_args()
  print(evaluate(
    args.model, args.output_dir, args.episodes, algo=args.algo,
    sequence=args.sequence, hit_radius=args.hit_radius,
    random_start=not args.fixed_start and args.start_pos is None,
    start_pos=args.start_pos,
  ))
