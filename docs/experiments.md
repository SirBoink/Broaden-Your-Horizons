# Experiments

Run from the repository root in `hrl`. Prepare archived records first. Use a new output directory for every changed configuration. The commands below can take more than five minutes; launch them manually.

## Discovery and historical transfer

```bash
python -m analysis.mechanism_experiment --stage preflight --output results/discovery --device cpu
python -m analysis.mechanism_experiment --stage audit --output results/discovery --device cpu
python -m analysis.mechanism_experiment --stage historical --output results/discovery --device cpu
```

The audit uses original ten-feature recordings, 40 discounts and frozen within-band reference distributions. Historical transfer restores the accepted base, reacher and selector; it does not refit them.

## Independent base policies and counterfactual handoffs

```bash
python -m scripts.evaluate_landmark_handoffs --output results/handoffs
python -m scripts.evaluate_hierarchy_geometry --stage run --output results/evaluation_seeds
python -m scripts.screen_landmark_evidence --replay --output results/local_replay --per-cell 6
```

Counterfactual branches restore actual invocation states. Reflected-waypoint exclusions and all policy results remain in the output. Evaluation seeds vary starting contexts, not trained policy identities.

## Equal-budget selectors and additional controls

```bash
python -m scripts.run_overnight_claims --output results/controls --hours 4.75
```

This resumable batch includes matched human profiles, alternate-sequence discovery, replay, state interventions and four option libraries with three selector seeds. Runtime limits preserve partial jobs and failures. Final checkpoints, not outcome-selected best checkpoints, are evaluated.

## New controller training

```bash
python -m SAC.changed.train --run-dir results/base_policy --timesteps 1000000 --seed 7 --sequence 0 3 7 5 6 1 2 4
python -m SAC.reacher train --run-path results/option_reacher --timesteps 1000000 --seed 7
python -m SAC.high_level train --run-path results/selector --options-path data/discovery/options/off_target_band_options.json --reacher-checkpoint checkpoints/option_reacher/best_model.zip --base-checkpoint checkpoints/base_policy/best_model.zip --timesteps 1000000 --seed 7 --free-for-all
```

These commands create fresh fits using the retained implementation. Frozen checkpoints provide the historical reference; exact selector refitting would require additional training metadata. Consult `--help` for each command and inspect `config/protocols/selector.json` before choosing a new training protocol.
