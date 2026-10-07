"""Unpack the archived trajectory records needed for simulation and SF fitting."""
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

from rich.progress import track

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    for name in ("data/neural/episodes", "data/handoffs/branches"):
        directory = ROOT / name
        archive = directory.with_suffix(".zip")
        if not archive.is_file():
            raise FileNotFoundError(f"Missing {archive}; run git lfs pull after cloning")
        with zipfile.ZipFile(archive) as source:
            for entry in track(source.infolist(), description=f"Preparing {directory.name}"):
                target = (directory / entry.filename).resolve()
                if not target.is_relative_to(directory.resolve()) or entry.is_dir():
                    raise ValueError(f"Invalid archive entry: {entry.filename}")
                if target.exists():
                    if target.stat().st_size == entry.file_size and zlib.crc32(target.read_bytes()) == entry.CRC:
                        continue
                    raise ValueError(f"Existing file differs from archive: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(source.read(entry))
                temporary.replace(target)
    manifest = json.loads((ROOT / "data/neural/manifest.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "config/neural_split.json").read_text(encoding="utf-8"))
    selected = set(split["train"] + split["validation"] + split["test"])
    for row in track(manifest["files"], description="Checking neural episode hashes"):
        if row["episode_id"] in selected:
            path = ROOT / "data/neural" / row["filename"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Episode checksum mismatch: {path}")


if __name__ == "__main__":
    prepare()
