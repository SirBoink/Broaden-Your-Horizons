# Broaden Your Horizons

Multiscale predictions of fingertip motion and muscle activation identify candidate intermediate reaches. A frozen SAC base policy and reacher execute them; a semi-Markov Double DQN selects when and where to intervene. Saved experiments compare sequence transfer, local handoffs and human reaching profiles.

The repository accompanies **Broaden Your Horizons**. The historical 10 ms Euler panel contains 240 matched trials per controller: base completes 153, single discount 169, and DDQN with A–E 204. Complete trial panels and supporting validation records accompany these results.

| Folder | Contents |
| --- | --- |
| `config/` | Shared parameters, function defaults, saved protocols and fixed numerical values |
| `formulas.py` | Predictive returns, band changes, discovery rules and SMDP targets |
| `SAC/` | Reaching environments, option execution and controller training |
| `analysis/` | Discovery audit, numerical integration, SF fitting and human analysis |
| `scripts/` | Evaluation, summaries, figure generation and verification |
| `data/` | Complete trial panels, raw trajectories and human comparison data |
| `checkpoints/` | Frozen base policies, option reacher and accepted selector |
| `figures/` | Presentation plots, explanatory diagrams and recorded animation |
| `presentation/` | Final presentation and slide-to-source index |

## Setup

Use the existing `hrl` environment, or create one with Python 3.10:

```bash
mamba create -n hrl -c conda-forge python=3.10
mamba activate hrl
uv pip install -r requirements.txt
git lfs install
git lfs pull
python -m scripts.prepare_data
```

The two trajectory archives are expanded locally and excluded from Git. Run commands from the repository root.

## Reproduce saved results

```bash
python -m scripts.verify_repository
python -m scripts.verify_repository --physics
python -m scripts.summarize_results
python -m scripts.reproduce_figures
```

Summaries go to `results/summary.json`; regenerated figures go to `results/figures/` and `results/curvature/`. Original presentation assets remain in `figures/`. Plot exports use 600 dpi PNG. These commands do not train controllers. The optional physics check requires the prepared trajectory archives.

## Parameters and equations

Edit `config/parameters.json` for shared discovery thresholds, reward weights, module constants and function defaults. `config/fixed_values.json` lists remaining fixed numerical literals by source location, including dimensions, indices and figure settings. `config/protocols/` preserves saved experiment settings; changing current parameters does not change historical checkpoints or records. Equations and units are in [docs/formulas.md](docs/formulas.md).

Long experiment commands are in [docs/experiments.md](docs/experiments.md). They display Rich progress and write to new result directories. Frozen-checkpoint evaluations and saved-record analyses provide the reproducible reference. New training uses the retained implementation; exact historical refitting would require additional selector reward metadata.

## Scope and sources

The trained hierarchy improves sequence completion in the original simulator. Supporting studies explore starting conditions, base policies, integration settings and local handoffs. Human profile and curvature comparisons provide exploratory behavioral context. Broader generalization and physiological interpretation remain topics for further validation. See [docs/results.md](docs/results.md).

Human data and published models: Cuevas Rivera and Kiebel, *Behavioral evidence for the hierarchical execution of sequential movements*, DOI [10.1038/s44271-026-00436-5](https://doi.org/10.1038/s44271-026-00436-5); model source [10.6084/m9.figshare.31169632](https://doi.org/10.6084/m9.figshare.31169632). Original attribution and source metadata are retained in `data/human/published_models/`. Third-party terms are summarized in [docs/attribution.md](docs/attribution.md).
