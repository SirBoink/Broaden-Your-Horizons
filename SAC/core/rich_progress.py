
from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("SAC/core/rich_progress.py", {})
DEFAULTS = SETTINGS["defaults"].get("SAC/core/rich_progress.py", {})
import time
import csv
from collections import deque
from pathlib import Path

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from stable_baselines3.common.callbacks import BaseCallback


class RichProgressCallback(BaseCallback):
  def __init__(self, loss_path: Path | None = None, verbose: int = DEFAULTS["RichProgressCallback.__init__"]["verbose"], algo: str = DEFAULTS["RichProgressCallback.__init__"]["algo"]):
    super().__init__(verbose)
    if algo not in {"ppo", "sac"}:
      raise ValueError("algo must be 'ppo' or 'sac'")
    self.algo = algo
    self.loss_path = loss_path
    self.last_update = -1
    self.episodes = 0
    self.successes = deque(maxlen=100)
    self.component_totals = {"progress": 0.0, "travel": 0.0, "target": 0.0, "completion": 0.0}
    self.last_components = self.component_totals.copy()
    self.last_return = None
    self.last_length = None
    self.eval_success = None
    self.best_eval = None
    self.eval_count = 0
    self.eval_streak = 0
    self.status = "training"

  def _init_callback(self):
    self.started = time.perf_counter()
    self.console = Console()
    self.progress = Progress(
      TextColumn("[progress.description]{task.description}"), BarColumn(),
      TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
      console=self.console, expand=True,
    )
    self.task_id = self.progress.add_task(f"{self.algo.upper()} training", total=self.model._total_timesteps)
    if self.loss_path:
      self.loss_path.parent.mkdir(parents=True, exist_ok=True)
      with self.loss_path.open("w", newline="", encoding="utf-8") as file:
        columns = (
          ("step", "policy_loss", "value_loss", "entropy_loss")
          if self.algo == "ppo" else
          ("step", "actor_loss", "critic_loss", "ent_coef", "ent_coef_loss")
        )
        csv.writer(file).writerow(columns)
    self.live = Live(self._render(), console=self.console, refresh_per_second=4)
    self.live.start()

  @staticmethod
  def _scalar(value, default=0.0):
    try:
      return float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
      return default

  def _train_metric(self, name):
    value = self.model.logger.name_to_value.get(name)
    return "--" if value is None else f"{float(value):.3g}"

  def _record_loss(self):
    if not self.loss_path:
      return
    values = self.model.logger.name_to_value
    update = values.get("train/n_updates")
    names = (
      ("train/policy_gradient_loss", "train/value_loss", "train/entropy_loss")
      if self.algo == "ppo" else
      ("train/actor_loss", "train/critic_loss", "train/ent_coef", "train/ent_coef_loss")
    )
    losses = [values.get(name) for name in names]
    if update is None or int(update) == self.last_update or any(value is None for value in losses):
      return
    with self.loss_path.open("a", newline="", encoding="utf-8") as file:
      csv.writer(file).writerow((self.num_timesteps, *[float(value) for value in losses]))
    self.last_update = int(update)

  def set_eval(self, success, best, count, streak):
    self.eval_success, self.best_eval = success, best
    self.eval_count, self.eval_streak = count, streak
    self.refresh()

  def set_status(self, status):
    self.status = status
    self.refresh()

  def refresh(self):
    if hasattr(self, "live"):
      self.live.update(self._render())

  def _on_step(self) -> bool:
    self._record_loss()
    infos = self.locals.get("infos", [])
    dones = self.locals.get("dones", [])
    for index, info in enumerate(infos):
      for key, output in (("progress_reward", "progress"), ("travel_reward", "travel"),
                          ("path_reward", "travel"),
                          ("target_reward", "target"), ("completion_reward", "completion")):
        self.component_totals[output] += self._scalar(info.get(key, 0.0))
      done = bool(np.asarray(dones[index]).reshape(-1)[0]) if len(dones) > index else False
      if done:
        episode = info.get("episode", {})
        self.episodes += 1
        self.last_return = self._scalar(episode.get("r"), self.last_return or 0.0)
        self.last_length = self._scalar(episode.get("l"), self.last_length or 0.0)
        self.successes.append(self._scalar(episode.get("is_success", 0.0)))
        self.last_components = self.component_totals.copy()
        self.component_totals = {"progress": 0.0, "travel": 0.0, "target": 0.0, "completion": 0.0}
    self.progress.update(self.task_id, completed=self.num_timesteps)
    self.live.update(self._render())
    return True

  def _on_training_end(self):
    self._record_loss()

  def _render(self):
    elapsed = max(time.perf_counter() - getattr(self, "started", time.perf_counter()), 1e-6)
    fps = self.num_timesteps / elapsed
    recent = list(self.model.ep_info_buffer)
    mean_return = np.mean([item["r"] for item in recent]) if recent else None
    mean_success = np.mean(self.successes) if self.successes else None
    value = lambda item: "--" if item is None else f"{item:.3f}"
    table = Table.grid(expand=True, padding=(0, 1))
    for _ in range(4):
      table.add_column(no_wrap=True)
    rows = [
      ("Steps", f"{self.num_timesteps}/{self.model._total_timesteps}", "FPS", f"{fps:.1f}"),
      ("Episodes", str(self.episodes), "Status", self.status),
      ("Last return", value(self.last_return), "Mean return", value(mean_return)),
      ("Last length", value(self.last_length), "Success", value(mean_success)),
      ("Progress R", f"{self.last_components['progress']:.3f}", "Target R", f"{self.last_components['target']:.3f}"),
      ("Path R", f"{self.last_components['travel']:.3f}", "Completion R", f"{self.last_components['completion']:.3f}"),
      ("Eval success", value(self.eval_success), "Best eval", value(self.best_eval)),
      ("Eval streak", str(self.eval_streak), "Evaluations", str(self.eval_count)),
    ]
    if self.algo == "ppo":
      rows.extend([
        ("Policy loss", self._train_metric("train/policy_gradient_loss"), "Value loss", self._train_metric("train/value_loss")),
        ("Entropy loss", self._train_metric("train/entropy_loss"), "Clip fraction", self._train_metric("train/clip_fraction")),
        ("Approx KL", self._train_metric("train/approx_kl"), "", ""),
      ])
    else:
      rows.extend([
        ("Actor loss", self._train_metric("train/actor_loss"), "Critic loss", self._train_metric("train/critic_loss")),
        ("Entropy coefficient", self._train_metric("train/ent_coef"), "Entropy coefficient loss", self._train_metric("train/ent_coef_loss")),
      ])
    for row in rows:
      table.add_row(*row)
    return Group(self.progress, table)

  def close(self):
    if hasattr(self, "live"):
      self.live.stop()
