"""Float64 reference mathematics shared by simulation and human analyses."""
from __future__ import annotations

from config import PARAMETERS as SETTINGS

import numpy as np
from scipy.ndimage import label
from scipy.signal import find_peaks
from scipy.stats import t

DT = SETTINGS["discovery"]["command_dt_s"]
SEEDS = tuple(SETTINGS["discovery"]["evaluation_seeds"])
NEURAL = SETTINGS["neural"]


def discount_grid():
    cfg = SETTINGS["discovery"]
    peaks = np.geomspace(cfg["minimum_horizon_s"], cfg["maximum_horizon_s"], cfg["band_count"])
    ratio = peaks[1] / peaks[0]
    tau = DT * (ratio - 1) / (ratio * np.log(ratio)) * ratio ** np.arange(SETTINGS["discovery"]["band_count"] + 1)
    return np.exp(-DT / tau)


def returns(phi, gammas):
    phi, gammas = np.asarray(phi, dtype=np.float64), np.asarray(gammas, dtype=np.float64)
    if phi.ndim != 2 or not np.isfinite(phi).all():
        raise ValueError('Features must be a finite time-by-feature matrix')
    if gammas.ndim != 1 or not np.isfinite(gammas).all() or np.any((gammas <= 0) | (gammas >= 1)):
        raise ValueError('Discounts must be strictly between zero and one')
    result = np.empty((len(gammas), len(phi), phi.shape[1]))
    future = np.zeros((len(gammas), phi.shape[1]))
    for k in range(len(phi) - 1, -1, -1):
        future = phi[k] + gammas[:, None] * future
        result[:, k] = future
    return result


def cosine_change(vectors):
    a, b = vectors[:, :-1], vectors[:, 1:]
    norms = np.linalg.norm(vectors, axis=-1)
    denominator = norms[:, :-1] * norms[:, 1:]
    valid = denominator > np.finfo(np.float64).eps
    cosine = np.divide(np.sum(a * b, axis=-1), denominator,
                       out=np.zeros_like(denominator), where=valid)
    return np.where(valid, 1 - np.clip(cosine, -1, 1), np.nan)


def missing_mass(gammas, remaining):
    a, b = np.asarray(gammas[:-1]), np.asarray(gammas[1:])
    total = 1 / (1 - b) - 1 / (1 - a)
    rem = np.asarray(remaining)
    return (b[:, None] ** rem / (1 - b[:, None])
            - a[:, None] ** rem / (1 - a[:, None])) / total[:, None]


def scores(phi, mean, std, gammas=None, gate=True):
    gammas = discount_grid() if gammas is None else np.asarray(gammas)
    std = np.asarray(std)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError('Feature scales must be finite and positive')
    psi = returns((phi - mean) / std, gammas)
    bands = np.diff(psi, axis=0) / np.abs(np.diff(-np.log(gammas)))[:, None, None]
    multi, single = cosine_change(bands), cosine_change(psi)
    if gate:
        remaining = np.arange(len(phi) - 1, 0, -1)
        multi[missing_mass(gammas, remaining) > SETTINGS["discovery"]["maximum_missing_mass"]] = np.nan
        single[gammas[:, None] ** remaining > SETTINGS["discovery"]["maximum_missing_mass"]] = np.nan
    return multi, single


def reference(values):
    return [np.sort(row[np.isfinite(row)]) for row in values]


def percentiles(values, refs):
    result = np.full_like(values, np.nan, dtype=np.float64)
    for i, ref in enumerate(refs):
        valid = np.isfinite(values[i])
        if len(ref):
            result[i, valid] = (np.searchsorted(ref, values[i, valid], 'left')
                               + np.searchsorted(ref, values[i, valid], 'right')) / (2 * len(ref))
    return result


def events(percentile, phi, stop, goals, target, trial, minimum_bands=SETTINGS["discovery"]["minimum_adjacent_bands"]):
    """Original component/tie/index rules; exclusion occurs after discovery merging."""
    active = percentile[:, :max(0, stop)] >= SETTINGS["discovery"]["percentile_threshold"]
    labels, count = label(active, structure=np.ones((3, 3), dtype=int))
    output = []
    for component in range(1, count + 1):
        cells = np.argwhere(labels == component)
        bands = set(cells[:, 0])
        if not any(set(range(int(b), int(b) + minimum_bands)) <= bands for b in bands):
            continue
        band, column = max(cells, key=lambda v: (percentile[tuple(v)], -v[1], -v[0]))
        step = int(column) + 1
        output.append(dict(trial=int(trial), target=int(target), step=step,
                           phase=step / max(stop, 1), bands=sorted(map(int, bands)),
                           peak_band=int(band), peak_percentile=float(percentile[band, column]),
                           xy_m=phi[step, :2].tolist(),
                           nearest_target_m=float(np.linalg.norm(goals - phi[step, :2], axis=1).min())))
    return output


def same_event(a, b):
    overlap = set(a['bands']) & set(b['bands'])
    return (a['target'] == b['target'] and abs(a['phase'] - b['phase']) <= SETTINGS["discovery"]["event_phase_tolerance"]
            and any(set(range(v, v + 3)) <= overlap for v in overlap))


def single_candidates(score, trajectories, stops, goals):
    """Five actual, separated local maxima; no fabricated positions or outcome selection."""
    candidates = []
    for trial, (row, phi, stop) in enumerate(zip(score, trajectories, stops)):
        row = row[:stop]
        for column in find_peaks(np.where(np.isfinite(row), row, -np.inf))[0]:
            xy = phi[column + 1, :2]
            if np.linalg.norm(goals - xy, axis=1).min() > SETTINGS["discovery"]["target_exclusion_m"]:
                candidates.append(dict(xy_m=xy.tolist(), score=float(row[column]),
                                       trial=trial, step=int(column) + 1))
    selected = []
    for item in sorted(candidates, key=lambda v: (-v['score'], v['trial'], v['step'])):
        if all(np.linalg.norm(np.asarray(item['xy_m']) - old['xy_m']) > SETTINGS["discovery"]["candidate_separation_m"] for old in selected):
            selected.append(item)
        if len(selected) == SETTINGS["discovery"]["candidate_count"]:
            break
    return selected


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite effects must be handled explicitly')
    n = len(values)
    if not n:
        return dict(n=0, mean=None, ci95=None)
    half = float(t.ppf((1 + SETTINGS["statistics"]["confidence_level"]) / 2, n - 1) * values.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=float(values.mean()),
                ci95=[float(values.mean() - half), float(values.mean() + half)] if half is not None else None,
                replication_values=values.tolist())


def discounted_trace(trace, gamma=None):
  gamma = SETTINGS["modules"]["SAC/high_level.py"]["GAMMA"] if gamma is None else gamma
  if not trace:
    raise ValueError("trace must not be empty")
  return float(sum(float(item["reward"]) * gamma ** index for index, item in enumerate(trace)))


def double_dqn_target(online, target, next_states, masks, rewards, durations, done,
                      gamma=None):
  import torch
  gamma = SETTINGS["modules"]["SAC/high_level.py"]["GAMMA"] if gamma is None else gamma
  with torch.no_grad():
    online_values = online(next_states).masked_fill(~masks, -torch.inf)
    choices = online_values.argmax(dim=1)
    target_values = target(next_states).gather(1, choices[:, None]).squeeze(1)
    return rewards + torch.pow(torch.as_tensor(gamma, device=durations.device), durations) * target_values * (~done)


def ridge_fits(x,y):
    mean=x.mean(0);scale=np.maximum(x.std(0),NEURAL["normalization_floor"])
    z=np.column_stack((np.ones(len(x)),(x-mean)/scale))
    gram=z.T@z;cross=z.T@y
    fitted=[]
    for alpha in NEURAL["ridge_strengths"]:
        penalty=np.eye(z.shape[1])*alpha*len(z);penalty[0,0]=0
        fitted.append((mean,scale,np.linalg.solve(gram+penalty,cross)))
    return fitted

def ridge_predict(fitted,x):
    mean,scale,weights=fitted
    return np.column_stack((np.ones(len(x)),(x-mean)/scale))@weights

def principal_features(x,train_stop):
    """Ten components fitted on calibration-training frames only; same size as one SF head."""
    mean=x[:train_stop].mean(0);scale=np.maximum(x[:train_stop].std(0),NEURAL["normalization_floor"])
    z=(x[:train_stop]-mean)/scale
    _,vectors=np.linalg.eigh(z.T@z/len(z))
    basis=vectors[:,-NEURAL["readout_components"]:]
    return (x-mean)/scale@basis,(mean,scale,basis)
