"""Train SMDP Double DQN selector over frozen SAC base and reacher."""

from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("SAC/high_level.py", {})
DEFAULTS = SETTINGS["defaults"].get("SAC/high_level.py", {})

import argparse
import csv
import hashlib
import json
import random
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import gymnasium as gym
import motornet as mn
import torch
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from torch import nn

from .changed.env import StraightReachEnv, StraightReachGym, sequence as DEFAULT_SEQUENCE
from .options import available_choice_mask, execute_choice
from formulas import discounted_trace, double_dqn_target


GAMMA = PARAMS["GAMMA"]
STATE_SIZE = PARAMS["STATE_SIZE"]
REACHER_SIZE = PARAMS["REACHER_SIZE"]
HIDDEN_SIZE = PARAMS["HIDDEN_SIZE"]
REPLAY_CAPACITY = PARAMS["REPLAY_CAPACITY"]
WARMUP = PARAMS["WARMUP"]
BATCH_SIZE = PARAMS["BATCH_SIZE"]
UPDATE_FREQUENCY = PARAMS["UPDATE_FREQUENCY"]
EVAL_FREQUENCY = PARAMS["EVAL_FREQUENCY"]
DEFAULT_TIMESTEPS = PARAMS["DEFAULT_TIMESTEPS"]
DEFAULT_SEED = PARAMS["DEFAULT_SEED"]
DEFAULT_BASE = Path(__file__).resolve().parents[1] / "checkpoints/base_policy/best_model.zip"
DEFAULT_DELIBERATION_COST = PARAMS["DEFAULT_DELIBERATION_COST"]


def load_frozen_sac(path, observation_size, device=DEFAULTS["load_frozen_sac"]["device"]):
  """Load the project's standard SAC policies without unpickling metadata.

  Rebuild serialized classes/spaces with native objects. SB3 in hrl loads
  tensor state dictionaries with torch.load(weights_only=True).
  """
  with zipfile.ZipFile(path) as archive:
    data = json.loads(archive.read("data"))
  serialized = {key for key, value in data.items()
                if isinstance(value, dict) and ":serialized:" in value}
  allowed = {"policy_class", "observation_space", "action_space", "lr_schedule",
             "replay_buffer_class", "train_freq", "_last_obs", "_last_episode_starts",
             "_last_original_obs", "ep_info_buffer", "ep_success_buffer"}
  if serialized - allowed:
    raise ValueError(f"unsupported serialized SAC metadata: {sorted(serialized - allowed)}")
  rate = float(data["learning_rate"])
  replacements = dict.fromkeys(serialized)
  replacements.update({
    "policy_class": SACPolicy,
    "observation_space": gym.spaces.Box(-np.inf, np.inf, (observation_size,), dtype=np.float32),
    "action_space": gym.spaces.Box(-1., 1., (6,), dtype=np.float32),
    "train_freq": (1, "step"), "lr_schedule": lambda _: rate,
  })
  return SAC.load(str(path), device=device, custom_objects=replacements)


class HierarchyProgress:
  def __init__(self, total, option_count):
    self.total = int(total)
    self.option_count = int(option_count)
    self.steps = self.episodes = self.decisions = self.explored = 0
    self.base_decisions = self.option_decisions = self.updates = 0
    self.last_loss = None
    self.last_return = self.last_targets = None
    self.eval_success = self.best_success = None
    self.eval_targets = self.eval_count = 0
    self.status = "starting"
    self.diagnostics = {}

  def start(self):
    self.started = time.perf_counter()
    self.console = Console()
    self.progress = Progress(
      TextColumn("[progress.description]{task.description}"), BarColumn(),
      TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
      console=self.console, expand=True,
    )
    self.task_id = self.progress.add_task(
      f"SMDP selector ({self.option_count} options)", total=self.total,
    )
    self.live = Live(self._render(), console=self.console, refresh_per_second=4)
    self.live.start()

  def _refresh(self):
    self.live.update(self._render())

  def set_status(self, status):
    self.status = status
    self._refresh()

  def set_evaluation(self, evaluation, best):
    self.eval_success = evaluation["success"]
    self.eval_targets = evaluation["targets_completed"]
    self.best_success = None if best is None else best["success"]
    self.eval_count += 1
    self._refresh()

  def update(self, primitive_steps, episode, updates, loss):
    self.steps = int(primitive_steps)
    self.episodes += 1
    self.decisions += int(episode["high_level_decisions"])
    self.explored += int(episode["exploration_decisions"])
    self.base_decisions += int(episode["action_frequencies"].get(0, 0))
    self.option_decisions = self.decisions - self.base_decisions
    self.updates = int(updates)
    self.last_loss = loss
    self.last_return = float(episode["return"])
    self.last_targets = int(episode["targets_completed"])
    self.progress.update(self.task_id, completed=self.steps)
    self._refresh()

  def _render(self):
    elapsed = max(time.perf_counter() - getattr(self, "started", time.perf_counter()), 1e-6)
    fps = self.steps / elapsed
    explore_rate = 100. * self.explored / max(self.decisions, 1)
    base_rate = 100. * self.base_decisions / max(self.decisions, 1)
    value = lambda item: "--" if item is None else str(item)
    table = Table.grid(expand=True, padding=(0, 1))
    for _ in range(4):
      table.add_column(no_wrap=True)
    table.add_row("Steps", f"{self.steps}/{self.total}", "FPS", f"{fps:.1f}")
    table.add_row("Episodes", str(self.episodes), "Status", self.status)
    table.add_row("Decisions", str(self.decisions), "Updates", str(self.updates))
    table.add_row("Base choices", f"{self.base_decisions} ({base_rate:.1f}%)",
                  "Option choices", str(self.option_decisions))
    table.add_row("Exploration", f"{explore_rate:.1f}%", "Last targets", value(self.last_targets))
    table.add_row("Last return", value(self.last_return), "Selector loss", value(self.last_loss))
    table.add_row("Eval success", value(self.eval_success), "Best success", value(self.best_success))
    table.add_row("Hidden gradients", value(self.diagnostics.get("hidden_grad_norms")),
                  "Q state variation", value(self.diagnostics.get("q_state_variation")))
    table.add_row("Episode base / option steps",
                  f"{self.diagnostics.get('base_steps', '--')} / {self.diagnostics.get('option_steps', '--')}",
                  "Episode option timeouts", value(self.diagnostics.get("option_timeouts")))
    table.add_row("Eval targets", str(self.eval_targets), "Evaluations", str(self.eval_count))
    return Group(self.progress, table)

  def close(self):
    if hasattr(self, "live"):
      self.live.stop()


class QNetwork(nn.Module):
  def __init__(self, state_dim=STATE_SIZE, action_count=DEFAULTS["QNetwork.__init__"]["action_count"]):
    super().__init__()
    if not isinstance(state_dim, int) or state_dim < 1 or not isinstance(action_count, int) or action_count < 1:
      raise ValueError("state_dim and action_count must be positive integers")
    self.state_dim = state_dim
    self.action_count = action_count
    self.net = nn.Sequential(
      nn.Linear(state_dim, HIDDEN_SIZE), nn.ReLU(),
      nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
      nn.Linear(HIDDEN_SIZE, action_count),
    )

  def forward(self, state):
    return self.net(state)


def epsilon(step, total):
  if total <= 0:
    raise ValueError("total must be positive")
  cfg = SETTINGS["training"]
  return cfg["epsilon_end"] + (cfg["epsilon_start"] - cfg["epsilon_end"]) * max(1. - float(step) / (cfg["epsilon_fraction"] * total), 0.)


def transfer_sequence(rng, heldout):
  while True:
    sequence = (DEFAULT_SEQUENCE[0], *map(int, rng.permutation(DEFAULT_SEQUENCE[1:])))
    if sequence not in heldout:
      return sequence


def masked_argmax(values, mask):
  values, mask = np.asarray(values, dtype=float), np.asarray(mask, dtype=bool)
  if values.shape != mask.shape or values.ndim != 1 or not mask.any() or not np.isfinite(values).all():
    raise ValueError("invalid masked Q values")
  return int(np.argmax(np.where(mask, values, -np.inf)))


def choose_action(values, mask, exploration, rng, free_for_all=DEFAULTS["choose_action"]["free_for_all"]):
  mask = np.asarray(mask, dtype=bool)
  if rng.random() >= exploration:
    return masked_argmax(values, mask), False
  if free_for_all:
    choices = np.flatnonzero(mask)
    if not len(choices):
      raise ValueError("no available selector choices")
    return int(rng.choice(choices)), True
  options = np.flatnonzero(mask[1:]) + 1
  if mask[0] and len(options):
    if rng.random() < .5:
      return 0, True
    return int(rng.choice(options)), True
  choices = np.flatnonzero(mask)
  if not len(choices):
    raise ValueError("no available selector choices")
  return int(rng.choice(choices)), True








def sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as source:
    for block in iter(lambda: source.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def load_option_set(path):
  data = json.loads(Path(path).read_text(encoding="utf-8"))
  if not isinstance(data, dict) or not isinstance(data.get("options"), list):
    raise ValueError("options file must contain options list")
  options = data["options"]
  if not options:
    raise ValueError("options file must contain at least one option")
  if [item.get("choice") for item in options] != list(range(1, len(options) + 1)):
    raise ValueError("option choices must be consecutive starting at one")
  return data


def _validate_policy(model, shape, name):
  if tuple(model.observation_space.shape) != shape or tuple(model.action_space.shape) != (6,):
    raise ValueError(f"{name} checkpoint must map {shape[0]} observations to 6 actions")


def make_env(sequence=DEFAULT_SEQUENCE, random_start=DEFAULTS["make_env"]["random_start"], max_episode_steps=DEFAULTS["make_env"]["max_episode_steps"],
             action_noise=DEFAULTS["make_env"]["action_noise"], obs_noise=DEFAULTS["make_env"]["obs_noise"], path_penalty_scale=DEFAULTS["make_env"]["path_penalty_scale"]):
  effector = mn.effector.RigidTendonArm26(
    muscle=mn.muscle.RigidTendonHillMuscle(), timestep=.01,
  )
  task = StraightReachEnv(
    effector, sequence, hit_radius=.02, max_episode_steps=max_episode_steps,
    differentiable=False, action_noise=action_noise, obs_noise=obs_noise,
    random_start=random_start, shuffle_sequence=False,
    path_penalty_scale=path_penalty_scale,
  )
  return StraightReachGym(task)


def _base_selector(action_count):
  selector = QNetwork(STATE_SIZE, action_count)
  for parameter in selector.parameters():
    parameter.data.zero_()
  return selector


def _training_networks(action_count, device, init_checkpoint=None):
  online = QNetwork(STATE_SIZE, action_count).to(device)
  target = QNetwork(STATE_SIZE, action_count).to(device)
  if init_checkpoint is not None:
    online.load_state_dict(load_checkpoint(init_checkpoint, device).state_dict())
  target.load_state_dict(online.state_dict())
  return online, target


def _result_score(result):
  return (
    int(result["success"]), int(result["targets_completed"]),
    -int(result["primitive_steps"]), float(result["return"]),
  )


def run_episode(env, base, reacher, options, selector, rng, *, step=DEFAULTS["run_episode"]["step"],
                total=DEFAULTS["run_episode"]["total"], training=DEFAULTS["run_episode"]["training"], reset_seed=DEFAULT_SEED,
                sequence=DEFAULT_SEQUENCE, capture_trajectory=DEFAULTS["run_episode"]["capture_trajectory"],
                free_for_all=DEFAULTS["run_episode"]["free_for_all"], deliberation_cost=DEFAULT_DELIBERATION_COST,
                primitive_budget=None):
  observation, info = env.reset(seed=reset_seed, options={
    "sequence": sequence, "deterministic": True,
  })
  if capture_trajectory:
    fingertip_path = [np.asarray(info["states"]["fingertip"]).reshape(-1, 2)[0].copy()]
    target_path = [int(np.argmax(observation[2:10]))]
    option_gamma_path = [np.nan]
    option_choice_path = [0]
    cumulative_return = [0.]
  transitions, decisions, reward_sum = [], 0, 0.
  frequencies, reasons = Counter(), Counter()
  exploration_count = 0
  decision_rows = []
  while True:
    mask = available_choice_mask(info, options)
    with torch.no_grad():
      q_values = selector(torch.as_tensor(
        observation[None], dtype=torch.float32, device=next(selector.parameters()).device,
      )).cpu().numpy()[0]
    choice, explored = choose_action(
      q_values, mask, epsilon(step, total) if training else 0., rng,
      free_for_all=free_for_all,
    )
    result = execute_choice(
      choice, observation, info, env, base, reacher, options,
      base_deterministic=True,
      max_steps=None if primitive_budget is None else int(primitive_budget - step),
    )
    if capture_trajectory:
      option_gamma = np.nan if choice == 0 else float(options["options"][choice - 1].get("gamma", np.nan))
      for item in result["trace"]:
        fingertip_path.append(np.asarray(
          item["next_info"]["states"]["fingertip"]
        ).reshape(-1, 2)[0].copy())
        target_path.append(int(np.argmax(item["next_observation"][2:10])))
        option_gamma_path.append(option_gamma)
        option_choice_path.append(choice)
        cumulative_return.append(cumulative_return[-1] + item["reward"])
    if result["duration"] < 1:
      raise ValueError("zero-duration selector transition")
    next_observation = np.asarray(result["final_observation"], dtype=np.float32)
    done = bool(result["terminated"] or result["truncated"])
    next_mask = np.ones(len(options["options"]) + 1, dtype=bool) if done else available_choice_mask(result["final_info"], options)
    discounted = discounted_trace(result["trace"]) - float(deliberation_cost)
    transition_step = step
    transitions.append((
      np.asarray(observation, dtype=np.float32), choice,
      discounted, result["duration"],
      next_observation, next_mask, done,
    ))
    decision_rows.append({
      "primitive_step": transition_step, "choice": choice,
      "option_id": result["option_id"],
      "target_index": int(np.asarray(info["sequence_index"]).reshape(-1)[0]),
      "exploration": bool(explored),
      "epsilon": epsilon(transition_step, total) if training else 0.,
      "duration": result["duration"], "reward_sum": result["reward_sum"],
      "discounted_reward": discounted, "reason": result["reason"],
    })
    decisions += 1
    exploration_count += int(explored)
    reward_sum += result["reward_sum"]
    frequencies[choice] += 1
    reasons[result["reason"]] += 1
    step += result["duration"]
    observation, info = next_observation, result["final_info"]
    if done or (primitive_budget is not None and step >= primitive_budget):
      break
  finished = bool(np.asarray(info.get("finished", False)).reshape(-1)[0])
  sequence_index = int(np.asarray(info.get("sequence_index", 0)).reshape(-1)[0])
  episode = {
    "success": finished,
    "targets_completed": sequence_index + int(finished),
    "return": float(reward_sum),
    "primitive_steps": int(env.task.elapsed_steps),
    "high_level_decisions": decisions,
    "exploration_decisions": exploration_count,
    "action_frequencies": dict(frequencies),
    "option_termination_reasons": dict(reasons),
    "_decision_rows": decision_rows,
  }
  if capture_trajectory:
    episode["fingertip_xy"] = np.asarray(fingertip_path, dtype=np.float32)
    episode["target_index"] = np.asarray(target_path, dtype=np.int8)
    episode["option_gamma"] = np.asarray(option_gamma_path, dtype=np.float32)
    episode["option_choice"] = np.asarray(option_choice_path, dtype=np.int32)
    episode["cumulative_return"] = np.asarray(cumulative_return, dtype=np.float64)
  return transitions, episode, step


def _update(online, target, optimizer, replay, device, diagnostics=None):
  states, actions, rewards, durations, next_states, masks, done = zip(*random.sample(replay, BATCH_SIZE))
  states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
  next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
  actions = torch.as_tensor(actions, dtype=torch.long, device=device)
  rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
  durations = torch.as_tensor(durations, dtype=torch.float32, device=device)
  masks = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)
  done = torch.as_tensor(done, dtype=torch.bool, device=device)
  all_values = online(states)
  values = all_values.gather(1, actions[:, None]).squeeze(1)
  targets = double_dqn_target(online, target, next_states, masks, rewards, durations, done)
  loss = torch.nn.functional.smooth_l1_loss(values, targets)
  optimizer.zero_grad(); loss.backward()
  if diagnostics is not None:
    diagnostics.update(
      hidden_grad_norms=[float(torch.cat((layer.weight.grad.flatten(), layer.bias.grad)).norm())
                         for layer in (online.net[0], online.net[2])],
      q_state_variation=float(all_values.detach().std(dim=0, unbiased=False).mean()),
    )
    if not np.isfinite([loss.item(), diagnostics["q_state_variation"],
                        *diagnostics["hidden_grad_norms"]]).all():
      raise RuntimeError("Non-finite selector learning diagnostics")
    if diagnostics["q_state_variation"] == 0 and not any(diagnostics["hidden_grad_norms"]):
      raise RuntimeError("Selector has zero state variation and zero hidden gradients")
  grad_norm = torch.nn.utils.clip_grad_norm_(online.parameters(), SETTINGS["training"]["selector_gradient_clip"]).item()
  optimizer.step()
  return loss.item(), values.mean().item(), values.max().item(), grad_norm


def train(run_path, options_path, reacher_checkpoint, *, base_checkpoint=DEFAULT_BASE,
          timesteps=DEFAULT_TIMESTEPS, seed=DEFAULT_SEED, device=DEFAULTS["train"]["device"],
          free_for_all=DEFAULTS["train"]["free_for_all"], transfer_training=DEFAULTS["train"]["transfer_training"], heldout_sequences=tuple(DEFAULTS["train"]["heldout_sequences"]),
          deliberation_cost=DEFAULT_DELIBERATION_COST, env_factory=None,
          init_checkpoint=None):
  run_path = Path(run_path)
  if run_path.exists():
    raise FileExistsError(run_path)
  for path in (base_checkpoint, reacher_checkpoint, options_path):
    if not Path(path).is_file():
      raise FileNotFoundError(path)
  if init_checkpoint is not None:
    saved = load_checkpoint(init_checkpoint, device).config
    for path in (base_checkpoint, reacher_checkpoint, options_path):
      if saved["artifact_hashes"].get(str(Path(path).resolve())) != sha256(path):
        raise ValueError(f"warm-start artifact differs: {path}")
  options = load_option_set(options_path)
  base = load_frozen_sac(base_checkpoint, STATE_SIZE, device=device)
  reacher = load_frozen_sac(reacher_checkpoint, REACHER_SIZE, device=device)
  _validate_policy(base, (STATE_SIZE,), "base")
  _validate_policy(reacher, (REACHER_SIZE,), "reacher")
  run_path.mkdir(parents=True)
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
  rng = np.random.default_rng(seed)
  env = make_env(random_start=transfer_training) if env_factory is None else env_factory()
  eval_env = make_env() if transfer_training else env
  action_count = len(options["options"]) + 1
  online, target = _training_networks(action_count, device, init_checkpoint)
  optimizer = torch.optim.Adam(online.parameters(), lr=SETTINGS["training"]["selector_learning_rate"])
  display = HierarchyProgress(timesteps, len(options["options"]))
  display.start()
  config = {
    "state_dim": STATE_SIZE, "option_count": len(options["options"]),
    "action_count": action_count, "hidden_units": [HIDDEN_SIZE, HIDDEN_SIZE],
    "timesteps": int(timesteps), "seed": int(seed), "device": device,
    "base_checkpoint": str(Path(base_checkpoint).resolve()),
    "reacher_checkpoint": str(Path(reacher_checkpoint).resolve()),
    "options_path": str(Path(options_path).resolve()),
    "sequence": list(DEFAULT_SEQUENCE), "discount": GAMMA,
    "deliberation_cost": float(deliberation_cost),
    "replay_capacity": REPLAY_CAPACITY, "warmup": WARMUP, "batch_size": BATCH_SIZE,
    "optimizer": f'Adam {SETTINGS["training"]["selector_learning_rate"]}', "target_sync_updates": UPDATE_FREQUENCY,
    "initialization": "PyTorch Linear defaults; target copied from online",
    "acceptance": "all eight targets completed; accepted.pt only saved on success",
    "epsilon": {"start": SETTINGS["training"]["epsilon_start"], "end": SETTINGS["training"]["epsilon_end"], "fraction": SETTINGS["training"]["epsilon_fraction"],
                "exploration_family_balance": "all_available_uniform" if free_for_all else "base_vs_options_50_50"},
    "transfer_training": transfer_training,
    "heldout_sequences": [list(sequence) for sequence in heldout_sequences],
    "environment": {"max_episode_steps": 800, "option_timeout_steps": 30,
                     "termination_radius_m": .02, "unrestricted_options": True},
    "artifact_hashes": {str(Path(path).resolve()): sha256(path) for path in (
      base_checkpoint, reacher_checkpoint, options_path,
    )},
  }
  if init_checkpoint is not None:
    config["initialization"] = "Saved DDQN weights; fresh optimizer, replay and exploration schedule"
    config["init_checkpoint"] = str(Path(init_checkpoint).resolve())
    config["init_checkpoint_sha256"] = sha256(init_checkpoint)
  (run_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
  display.set_status("baseline")
  baseline = _base_selector(action_count).to(device)
  _, control, _ = run_episode(eval_env, base, reacher, options, baseline, rng,
                              total=timesteps, deliberation_cost=deliberation_cost)
  (run_path / "control.json").write_text(json.dumps(control, indent=2), encoding="utf-8")
  torch.save({"model": baseline.state_dict(), "config": config}, run_path / "baseline.pt")
  display.set_status("training")
  replay = []
  replay_position = 0
  updates = 0
  primitive = 0
  best = None
  last_loss = None
  next_eval = EVAL_FREQUENCY
  with (run_path / "decisions.csv").open("x", newline="", encoding="utf-8") as decisions_file, (run_path / "evaluations.csv").open("x", newline="", encoding="utf-8") as evaluations_file, (run_path / "learning.csv").open("x", newline="", encoding="utf-8") as learning_file:
    learning_writer = csv.DictWriter(learning_file, fieldnames=[
      "primitive_step", "updates", "loss", "hidden_grad_norms", "q_state_variation",
      "targets_completed", "base_steps", "option_steps", "option_timeouts",
    ])
    learning_writer.writeheader()
    decision_writer = csv.writer(decisions_file)
    decision_writer.writerow(["primitive_step", "choice", "option_id", "exploration", "epsilon", "duration", "reward_sum", "discounted_reward", "reason"])
    evaluation_writer = csv.DictWriter(evaluations_file, fieldnames=["primitive_step", "success", "targets_completed", "return", "primitive_steps", "high_level_decisions", "action_frequencies", "option_termination_reasons"])
    evaluation_writer.writeheader()
    while primitive < int(timesteps):
      sequence = (transfer_sequence(rng, heldout_sequences) if transfer_training
                  else DEFAULT_SEQUENCE)
      transitions, episode, new_primitive = run_episode(
        env, base, reacher, options, online, rng, step=primitive,
        total=timesteps, training=True, free_for_all=free_for_all,
        sequence=sequence,
        reset_seed=seed + display.episodes + 1 if transfer_training or env_factory is not None else DEFAULT_SEED,
        deliberation_cost=deliberation_cost,
        primitive_budget=int(timesteps),
      )
      diagnostics = {}
      for index, transition in enumerate(transitions):
        if len(replay) < REPLAY_CAPACITY:
          replay.append(transition)
        else:
          replay[replay_position] = transition
          replay_position = (replay_position + 1) % REPLAY_CAPACITY
        if len(replay) >= WARMUP and len(replay) >= BATCH_SIZE:
          loss, q_mean, q_max, grad_norm = _update(
            online, target, optimizer, replay, device,
            diagnostics if index == len(transitions) - 1 or updates == 0 else None,
          )
          last_loss = loss
          updates += 1
          if updates % UPDATE_FREQUENCY == 0:
            target.load_state_dict(online.state_dict())
      for row in episode["_decision_rows"]:
        decision_writer.writerow([
          row["primitive_step"], row["choice"], row["option_id"],
          row["exploration"], row["epsilon"], row["duration"],
          row["reward_sum"], row["discounted_reward"], row["reason"],
      ])
      primitive = new_primitive
      diagnostics.update(
        primitive_step=primitive, updates=updates, loss=last_loss,
        targets_completed=episode["targets_completed"],
        base_steps=sum(row["duration"] for row in episode["_decision_rows"] if row["choice"] == 0),
        option_steps=sum(row["duration"] for row in episode["_decision_rows"] if row["choice"] != 0),
        option_timeouts=episode["option_termination_reasons"].get("option_timeout", 0),
      )
      learning_writer.writerow(diagnostics)
      learning_file.flush()
      decisions_file.flush()
      display.diagnostics = diagnostics
      display.update(primitive, episode, updates, last_loss)
      if primitive >= next_eval or primitive >= int(timesteps):
        display.set_status("evaluating")
        _, evaluation, _ = run_episode(eval_env, base, reacher, options, online, rng,
                                       total=timesteps, deliberation_cost=deliberation_cost)
        evaluation = dict(evaluation)
        evaluation.pop("_decision_rows", None)
        evaluation["primitive_step"] = primitive
        evaluation_row = {key: evaluation[key] for key in evaluation_writer.fieldnames}
        evaluation_row["action_frequencies"] = json.dumps(evaluation["action_frequencies"])
        evaluation_row["option_termination_reasons"] = json.dumps(evaluation["option_termination_reasons"])
        evaluation_writer.writerow(evaluation_row)
        evaluations_file.flush()
        if best is None or _result_score(evaluation) > _result_score(best):
          best = evaluation
          torch.save({"model": online.state_dict(), "config": config}, run_path / "best.pt")
          if best["success"] and best["targets_completed"] == len(DEFAULT_SEQUENCE):
            torch.save({"model": online.state_dict(), "config": config}, run_path / "accepted.pt")
        display.set_evaluation(evaluation, best)
        display.set_status("training")
        while next_eval <= primitive:
          next_eval += EVAL_FREQUENCY
  torch.save({"model": online.state_dict(), "config": config}, run_path / "final.pt")
  env.close()
  if eval_env is not env:
    eval_env.close()
  display.set_status("complete")
  display.close()
  summary = {"control": control, "best": best, "updates": updates,
             "primitive_steps": primitive, "rejected": best is None or not best["success"]}
  summary["accepted_checkpoint"] = "accepted.pt" if (run_path / "accepted.pt").exists() else None
  summary["rejected"] = summary["accepted_checkpoint"] is None
  summary["baseline_checkpoint"] = "baseline.pt"
  summary["improves_baseline_steps"] = bool(
    not summary["rejected"] and best["primitive_steps"] < control["primitive_steps"]
  )
  (run_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  return summary


def load_checkpoint(path, device=DEFAULTS["load_checkpoint"]["device"]):
  checkpoint = torch.load(path, map_location=device, weights_only=True)
  config = checkpoint["config"]
  network = QNetwork(int(config["state_dim"]), int(config["action_count"])).to(device)
  network.load_state_dict(checkpoint["model"])
  network.eval()
  network.config = config
  return network


def build_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=("train",))
  parser.add_argument("--run-path", type=Path, required=True)
  parser.add_argument("--options-path", type=Path, required=True)
  parser.add_argument("--reacher-checkpoint", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
  parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--init-checkpoint", type=Path,
                      help="warm-start weights in a new run; optimizer and replay reset")
  parser.add_argument("--deliberation-cost", type=float, default=DEFAULT_DELIBERATION_COST,
                      help=f"deliberation cost subtracted per high-level decision (default: {DEFAULT_DELIBERATION_COST})")
  parser.add_argument("--free-for-all", action="store_true",
                      help="sample uniformly from all available choices during exploration")
  return parser


def main():
  args = build_parser().parse_args()
  if args.timesteps < 1:
    raise ValueError("--timesteps must be positive")
  summary = train(
    args.run_path, args.options_path, args.reacher_checkpoint,
    base_checkpoint=args.base_checkpoint, timesteps=args.timesteps,
    seed=args.seed, device=args.device, free_for_all=args.free_for_all,
    deliberation_cost=args.deliberation_cost,
    init_checkpoint=args.init_checkpoint,
  )
  print(f"Baseline: {summary['control']['primitive_steps']} steps; "
        f"accepted selector: {summary['accepted_checkpoint'] or 'none'}; "
        f"faster than baseline: {summary['improves_baseline_steps']}")
  if summary["rejected"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
