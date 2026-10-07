
from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("SAC/changed/train.py", {})
DEFAULTS = SETTINGS["defaults"].get("SAC/changed/train.py", {})
import argparse
import json
from pathlib import Path

import motornet as mn
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from SAC.core.env import target_hit_radius
from SAC.core.metadata import write_metadata
from SAC.core.rich_progress import RichProgressCallback

from .env import StraightReachEnv, StraightReachGym, sequence as default_sequence
from .evaluate import make_env as make_eval_env, run_evaluation


def make_training_env(seed: int, monitor_path: Path, sequence=default_sequence,
                      hit_radius: float = target_hit_radius, *,
                      random_start: bool = DEFAULTS["make_training_env"]["random_start"], shuffle_sequence: bool = DEFAULTS["make_training_env"]["shuffle_sequence"],
                      start_pos=None):
  def factory():
    effector = mn.effector.RigidTendonArm26(
      muscle=mn.muscle.RigidTendonHillMuscle(), timestep=0.01
    )
    env = StraightReachGym(StraightReachEnv(
      effector, sequence, hit_radius=hit_radius, differentiable=False,
      action_noise=0.0, obs_noise=0.0, random_start=random_start,
      shuffle_sequence=shuffle_sequence, start_pos=start_pos,
      path_penalty_scale=0.05,
    ))
    return Monitor(env, filename=str(monitor_path), info_keywords=("is_success",))
  factory.seed = seed
  return factory


class QualityEvalCallback(BaseCallback):
  def __init__(self, eval_freq: int, episodes: int, output_dir: Path,
               sequence, hit_radius: float, display=None,
               success_threshold: float = DEFAULTS["QualityEvalCallback.__init__"]["success_threshold"], path_ratio_threshold: float = DEFAULTS["QualityEvalCallback.__init__"]["path_ratio_threshold"],
               deviation_threshold: float = DEFAULTS["QualityEvalCallback.__init__"]["deviation_threshold"], patience: int = DEFAULTS["QualityEvalCallback.__init__"]["patience"], *,
               random_start: bool = DEFAULTS["QualityEvalCallback.__init__"]["random_start"], start_pos=None):
    super().__init__()
    self.eval_freq = eval_freq
    self.episodes = episodes
    self.output_dir = output_dir
    self.sequence = sequence
    self.hit_radius = hit_radius
    self.random_start = random_start
    self.start_pos = start_pos
    self.display = display
    self.success_threshold = success_threshold
    self.path_ratio_threshold = path_ratio_threshold
    self.deviation_threshold = deviation_threshold
    self.patience = patience
    self.best_score = (-1.0, -np.inf)
    self.best_success = -1.0
    self.history = []
    self.streak = 0
    self.stopped_early = False
    self.stop_reason = "completed timestep budget"

  def _init_callback(self):
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.eval_env = make_eval_env(
      self.sequence, self.hit_radius, random_start=self.random_start,
      start_pos=self.start_pos, path_penalty_scale=0.05,
    )

  def _evaluate(self, name: str, save_episodes: bool = False):
    if self.display:
      self.display.set_status("evaluating")
    episodes = run_evaluation(
      self.model, self.eval_env, self.output_dir / name, self.episodes, save_episodes
    )
    success = float(np.mean([episode["success"] for episode in episodes]))
    ratios = [episode["mean_path_ratio"] for episode in episodes
              if episode["mean_path_ratio"] is not None]
    deviations = [episode["max_deviation"] for episode in episodes
                  if episode["max_deviation"] is not None]
    path_ratio = float(np.mean(ratios)) if ratios else np.inf
    max_deviation = float(max(deviations)) if deviations else np.inf
    record = {
      "step": self.num_timesteps,
      "success_rate": success,
      "mean_path_ratio": None if not ratios else path_ratio,
      "max_deviation": None if not deviations else max_deviation,
      "episodes": episodes,
    }
    self.history.append(record)
    (self.output_dir / "history.json").write_text(
      json.dumps(self.history, indent=2), encoding="utf-8"
    )

    score = (success, -path_ratio)
    if score > self.best_score:
      self.best_score = score
      self.best_success = success
      self.model.save(str(self.output_dir.parent / "best_model"))
    quality_met = (
      success >= self.success_threshold
      and path_ratio <= self.path_ratio_threshold
      and max_deviation <= self.deviation_threshold
    )
    self.streak = self.streak + 1 if quality_met else 0
    if self.display:
      self.display.set_eval(success, self.best_success, len(self.history), self.streak)
      self.display.set_status("training")

  def _on_step(self):
    if self.n_calls % self.eval_freq:
      return True
    self._evaluate(f"step_{self.num_timesteps:08d}")
    if self.streak < self.patience:
      return True
    self.stopped_early = True
    self.stop_reason = "success and trajectory quality thresholds reached"
    return False

  def _on_training_end(self):
    self._evaluate("final", save_episodes=True)
    self.eval_env.close()
    if self.display:
      self.display.set_status("complete")
      self.display.close()


def train(run_dir: Path, total_timesteps: int = DEFAULTS["train"]["total_timesteps"], seed: int = DEFAULTS["train"]["seed"],
          eval_freq: int = DEFAULTS["train"]["eval_freq"], checkpoint_freq: int | None = DEFAULTS["train"]["checkpoint_freq"],
          eval_episodes: int = DEFAULTS["train"]["eval_episodes"], *, algo: str = DEFAULTS["train"]["algo"],
          sequence=default_sequence, hit_radius: float = target_hit_radius,
          random_start: bool = DEFAULTS["train"]["random_start"], shuffle_sequence: bool = DEFAULTS["train"]["shuffle_sequence"], start_pos=None,
          initial_model: Path | None = None):
  if initial_model is not None and not initial_model.is_file():
    raise FileNotFoundError(f"initial model not found: {initial_model}")
  run_dir.mkdir(parents=True, exist_ok=True)
  vec_env = DummyVecEnv([make_training_env(
    seed, run_dir / "monitor.csv", sequence, hit_radius,
    random_start=random_start, shuffle_sequence=shuffle_sequence, start_pos=start_pos,
  )])
  display = RichProgressCallback(run_dir / "loss_history.csv", algo=algo)
  evaluation = QualityEvalCallback(
    eval_freq, eval_episodes, run_dir / "evaluations", sequence, hit_radius, display,
    random_start=random_start, start_pos=start_pos,
  )
  callbacks = [display, evaluation]
  if checkpoint_freq:
    callbacks.insert(0, CheckpointCallback(
      save_freq=checkpoint_freq, save_path=str(run_dir / "checkpoints"),
      name_prefix=algo, verbose=0,
    ))
  model_class = SAC if algo == "sac" else PPO
  if initial_model is not None:
    model = model_class.load(str(initial_model), env=vec_env)
  else:
    if algo == "sac":
      model = SAC(
        "MlpPolicy", vec_env, **SETTINGS["training"]["sac"], seed=seed, verbose=0,
      )
    else:
      model = PPO(
        "MlpPolicy", vec_env, **SETTINGS["training"]["ppo"], seed=seed, verbose=0
      )
  model.learn(
    total_timesteps=total_timesteps, callback=callbacks,
    reset_num_timesteps=initial_model is None,
  )
  model.save(str(run_dir / "final_model"))
  write_metadata(
    run_dir, model, total_timesteps, seed, eval_freq, checkpoint_freq, eval_episodes,
    model.num_timesteps, 0.95, 3, evaluation.stopped_early, evaluation.stop_reason,
    algo=algo, sequence=sequence, hit_radius=hit_radius,
  )
  config_path = run_dir / "config.json"
  config = json.loads(config_path.read_text(encoding="utf-8"))
  config["reward"] = {
    "progress_scale": 1.0,
    "distance_scale": 0.1,
    "path_penalty_scale": 0.05,
    "target": 20.0,
    "completion": 200.0,
  }
  config["environment"].update({
    "max_episode_steps": 800,
    "arrival_speed": 0.1,
    "random_training_starts": random_start,
    "shuffled_training_sequence": shuffle_sequence,
    "start_pos_degrees": None if start_pos is None else [float(value) for value in start_pos],
    "initial_model": None if initial_model is None else str(initial_model),
  })
  config["observation_shape"] = list(model.observation_space.shape)
  config["observation_addition"] = "six current muscle activations"
  config["model_selection"] = {
    "success_rate": 0.95,
    "mean_path_ratio": 1.2,
    "max_deviation": 0.02,
    "patience": 3,
  }
  config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
  vec_env.close()
  return run_dir / "final_model.zip"


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-dir", type=Path, default=Path("runs/changed_sac_seed7"))
  parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
  parser.add_argument("--timesteps", type=int, default=1_000_000)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--eval-freq", type=int, default=10_000)
  parser.add_argument("--checkpoint-freq", type=int, default=50_000)
  parser.add_argument("--eval-episodes", type=int, default=10)
  parser.add_argument("--sequence", type=int, nargs="+", default=None,
                      help="fixed target sequence; omit to shuffle each training episode")
  parser.add_argument("--start-pos", type=float, nargs=2,
                      metavar=("SHOULDER_DEG", "ELBOW_DEG"),
                      help="fixed starting joint angles in degrees")
  parser.add_argument("--initial-model", type=Path,
                      help="continue training from saved model")
  parser.add_argument("--hit-radius", type=float, default=target_hit_radius)
  args = parser.parse_args()
  sequence = default_sequence if args.sequence is None else args.sequence
  print(train(
    args.run_dir, args.timesteps, args.seed, args.eval_freq,
    args.checkpoint_freq or None, args.eval_episodes, algo=args.algo,
    sequence=sequence, hit_radius=args.hit_radius,
    random_start=args.start_pos is None,
    shuffle_sequence=args.sequence is None,
    start_pos=args.start_pos,
    initial_model=args.initial_model,
  ))
