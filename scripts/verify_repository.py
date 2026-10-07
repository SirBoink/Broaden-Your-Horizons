"""Check source integrity, predictive equations and the saved presentation totals."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from rich.progress import track

import formulas
from scripts.summarize_results import summarize

ROOT = Path(__file__).resolve().parents[1]


def verify(physics=False):
    for path in track(sorted(ROOT.rglob("*.py")), description="Checking Python syntax"):
        if not any(part in (".git", "__pycache__") for part in path.parts):
            ast.parse(path.read_text(encoding="utf-8-sig"))
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for name, expected in track(manifest.items(), description="Checking research files"):
        path = ROOT / name
        assert path.is_file(), f"Missing {name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Changed {name}"
    gamma = np.array([.5, .8, .9])
    phi = np.arange(120, dtype=float).reshape(40, 3) / 100
    returns = formulas.returns(phi, gamma)
    np.testing.assert_allclose(returns[:, :-1], phi[:-1] + gamma[:, None, None] * returns[:, 1:])
    np.testing.assert_allclose(returns[:, -1], np.broadcast_to(phi[-1], (3, 3)))
    assert len(formulas.discount_grid()) == 40
    assert np.all(np.diff(formulas.discount_grid()) > 0)
    assert np.isnan(formulas.cosine_change(np.zeros((2, 3, 4)))).all()
    assert formulas.discounted_trace([{"reward": 1}, {"reward": 2}], gamma=.5) == 2
    x = np.arange(3, dtype=float)[:, None]
    for strength, fit in zip(formulas.NEURAL["ridge_strengths"], formulas.ridge_fits(x, x)):
        np.testing.assert_allclose(formulas.ridge_predict(fit, x), 1 + (x - 1) / (1 + strength))
    results = summarize()
    assert [results["full"][m]["successes"] for m in ("base", "single", "learned")] == [153, 169, 204]
    assert [results["short_transitions"][m]["successes"] for m in ("base", "single", "learned")] == [38, 48, 74]
    assert results["policy_replication"]["7"]["ddqn"]["successes"] == 208
    assert results["demonstrations"]["A"]["option_segments"] == 122
    assert results["demonstrations"]["B"]["option_commands"] == 1116
    assert results["local_replay"]["0.0005"] == {"cases": 80, "better": 29, "tied": 38, "worse": 13}
    if physics:
        import torch
        from analysis.mechanism_checks import run_checks
        from analysis.mechanism_experiment import verified_selector
        from analysis.mechanism_validation import load_network

        assert all(run_checks(ROOT).values())
        assert verified_selector().config["action_count"] == 6
        network = load_network(ROOT / "data/neural/initial_model/best.pt")
        with torch.no_grad():
            prediction = network(torch.zeros((2, 30)))
        assert prediction.shape == (2, 23, 10) and torch.isfinite(prediction).all()
        online = lambda x: torch.tensor([[10., 20.], [30., 40.]])
        target = lambda x: torch.tensor([[1., 2.], [3., 4.]])
        value = formulas.double_dqn_target(online, target, torch.zeros((2, 1)),
            torch.tensor([[True, False], [True, True]]), torch.tensor([5., 6.]),
            torch.tensor([2, 3]), torch.tensor([False, True]), gamma=.5)
        torch.testing.assert_close(value, torch.tensor([5.25, 6.]))
    with zipfile.ZipFile(ROOT / "presentation/Broaden_Your_Horizons.pptx") as deck:
        assert deck.testzip() is None
        slides = json.loads((ROOT / "presentation/slides.json").read_text(encoding="utf-8"))
        assert len(slides) == 41
        for slide in slides:
            assert all((ROOT / image).is_file() for image in slide["figures"])
    print("Verified source files, equations, complete trial counts and 41 presentation slides.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", action="store_true", help="Also check the simulator and frozen networks; requires prepared archives")
    verify(parser.parse_args().physics)
