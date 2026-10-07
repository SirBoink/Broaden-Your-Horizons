# Configuration

| File | Meaning |
| --- | --- |
| `parameters.json` | Editable discovery, reward, module constants and function defaults, read at import |
| `fixed_values.json` | Remaining fixed numerical literals with exact source locations |
| `protocols/` | Saved experiment settings and checkpoint metadata; historical records |
| `heldout_panel.json` | Original transfer orders and initial postures |
| `neural_split.json` | Disjoint 3360/720/720 episode identifiers |
| `environment.json` | Python and package versions used for verification |

Default arguments are bound when a module loads. Restart Python after editing parameters. Array indices, algorithm identities and plot coordinates remain fixed and are indexed separately. Changed parameters require a fresh output directory and do not alter saved evidence.
