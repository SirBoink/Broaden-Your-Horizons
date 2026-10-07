"""Availability masks and execution of frozen intermediate-reaching options."""
from config import PARAMETERS as SETTINGS
import copy
from numbers import Integral, Real
import numpy as np
PARAMS = SETTINGS["modules"].get("SAC/options.py", {})
DEFAULTS = SETTINGS["defaults"].get("SAC/options.py", {})

def _option_xy(option):
  value = np.asarray(option.get("xy_m"), dtype=np.float32).reshape(-1)
  if value.shape != (2,) or not np.isfinite(value).all():
    raise ValueError("option xy_m must contain two finite values")
  return value

def _fingertip(info):
  states = info.get("states", info)
  value = states.get("fingertip") if isinstance(states, dict) else None
  if value is None:
    raise KeyError("info must contain states['fingertip']")
  value = np.asarray(value, dtype=float).reshape(-1)
  if value.shape != (2,) or not np.isfinite(value).all():
    raise ValueError("fingertip must contain two finite values")
  return value

def _options(option_set):
  values = option_set.get("options") if isinstance(option_set, dict) else option_set
  if not isinstance(values, list):
    raise ValueError("option_set must contain list under 'options'")
  return values

def _radius(option, option_set):
  value = option.get("termination_radius_m")
  if value is None and isinstance(option_set, dict):
    value = option_set.get("termination_radius_m")
  if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
    raise ValueError("termination radius must be positive and finite")
  return float(value)

def _action(prediction):
  action = prediction[0] if isinstance(prediction, tuple) else prediction
  action = np.asarray(action, dtype=np.float32)
  if action.shape == (1, 6):
    action = action[0]
  if action.shape != (6,) or not np.isfinite(action).all():
    raise ValueError("policy action must be a finite six-dimensional vector")
  return action

def _reward(value):
  value = np.asarray(value, dtype=float)
  if not np.isfinite(value).all():
    raise ValueError("environment reward must be finite")
  return float(value.sum())

def _trace_item(observation, action, reward, next_observation, info, next_info,
                terminated, truncated, step_offset):
  return {
    "observation": np.asarray(observation, dtype=np.float32).copy(),
    "action": action.copy(),
    "reward": float(reward),
    "next_observation": np.asarray(next_observation, dtype=np.float32).copy(),
    "info": copy.deepcopy(info),
    "next_info": copy.deepcopy(next_info),
    "terminated": bool(terminated),
    "truncated": bool(truncated),
    "step_offset": int(step_offset),
  }

def _result(choice, option_id, duration, reward_sum, observation, info,
            terminated, truncated, reason, trace):
  return {
    "choice": int(choice),
    "option_id": option_id,
    "duration": int(duration),
    "reward_sum": float(reward_sum),
    "final_observation": np.asarray(observation, dtype=np.float32).copy(),
    "final_info": copy.deepcopy(info),
    "terminated": bool(terminated),
    "truncated": bool(truncated),
    "reason": reason,
    "trace": trace,
  }

def available_choice_mask(info, option_set):
  options = _options(option_set)
  choices = [option.get("choice") for option in options]
  if choices != list(range(1, len(options) + 1)):
    raise ValueError("option choices must be consecutive integers starting at one")
  mask = np.ones(len(options) + 1, dtype=bool)
  fingertip = _fingertip(info)
  for option in options:
    choice = option.get("choice")
    if not isinstance(choice, Integral) or isinstance(choice, bool) or not 1 <= choice <= len(options):
      raise ValueError("option choices must be consecutive integers starting at one")
    if np.linalg.norm(fingertip - _option_xy(option)) <= _radius(option, option_set):
      mask[int(choice)] = False
  return mask

def execute_choice(choice, observation, info, env, base_policy, reacher,
                   option_set, *, base_deterministic=DEFAULTS["execute_choice"]["base_deterministic"], max_steps=None):
  observation = np.asarray(observation, dtype=np.float32)
  if observation.shape != (30,) or not np.isfinite(observation).all():
    raise ValueError("observation must be a finite 30-dimensional vector")
  options = _options(option_set)
  if not isinstance(choice, Integral) or isinstance(choice, bool):
    raise ValueError("choice must be an integer")
  choice = int(choice)
  if max_steps is not None and (not isinstance(max_steps, Integral) or max_steps < 1):
    raise ValueError("max_steps must be a positive integer")
  if choice == BASE_CHOICE:
    if base_policy is None:
      raise ValueError("base_policy required for BASE choice")
    action = _action(base_policy.predict(observation, deterministic=base_deterministic))
    next_observation, reward, terminated, truncated, next_info = env.step(action)
    reward = _reward(reward)
    return _result(choice, None, 1, reward, next_observation, next_info,
                   terminated, truncated, "base_step", [_trace_item(
                     observation, action, reward, next_observation, info, next_info,
                     terminated, truncated, 0)])
  if not 1 <= choice <= len(options):
    raise ValueError(f"unknown choice {choice}")
  option = next((item for item in options if item.get("choice") == choice), None)
  if option is None or reacher is None:
    raise ValueError("reacher required for valid option choice")
  xy = _option_xy(option)
  radius = _radius(option, option_set)
  timeout = option.get("timeout_steps", option_set.get("timeout_steps")
                     if isinstance(option_set, dict) else None)
  if timeout is None:
    timeout = TIMEOUT_STEPS
  timeout = min(int(timeout), TIMEOUT_STEPS)
  if max_steps is not None:
    timeout = min(timeout, int(max_steps))
  if not isinstance(timeout, Integral) or isinstance(timeout, bool) or timeout <= 0:
    raise ValueError("option timeout_steps must be a positive integer")
  if np.linalg.norm(_fingertip(info) - xy) <= radius:
    return _result(choice, option.get("id"), 0, 0., observation, info,
                   False, False, "already_in_region", [])

  current_observation, current_info = observation, info
  trace, reward_sum = [], 0.
  terminated = truncated = False
  reason = "option_timeout"
  for step_offset in range(int(timeout)):
    reacher_observation = np.concatenate((xy, current_observation[10:30]))
    if reacher_observation.shape != (22,):
      raise ValueError("reacher observation must have shape (22,)")
    action = _action(reacher.predict(reacher_observation, deterministic=True))
    next_observation, reward, terminated, truncated, next_info = env.step(action)
    reward = _reward(reward)
    reward_sum += reward
    trace.append(_trace_item(current_observation, action, reward, next_observation,
                             current_info, next_info, terminated, truncated, step_offset))
    current_observation = np.asarray(next_observation, dtype=np.float32)
    if current_observation.shape != (30,) or not np.isfinite(current_observation).all():
      raise ValueError("environment next observation must be a finite 30-dimensional vector")
    current_info = next_info
    if terminated:
      reason = "environment_terminated"
      break
    if truncated:
      reason = "environment_truncated"
      break
    if np.linalg.norm(_fingertip(current_info) - xy) <= radius:
      reason = "subgoal_reached"
      break
  return _result(choice, option.get("id"), len(trace), reward_sum,
                 current_observation, current_info, terminated, truncated, reason, trace)
