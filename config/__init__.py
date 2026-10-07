"""Research parameters shared by training, analysis and figure scripts."""
import json
from pathlib import Path

PARAMETERS = json.loads(Path(__file__).with_name("parameters.json").read_text(encoding="utf-8"))
if PARAMETERS["discovery"]["command_dt_s"] != PARAMETERS["protocol"]["command_dt"]:
    raise ValueError("Discovery and command timesteps must agree")
if not 0 < PARAMETERS["discovery"]["percentile_threshold"] < 1:
    raise ValueError("Discovery percentile must lie strictly between zero and one")
