"""Process every new target under a data root using a TOML config file.

Usage: python 02_batch_with_config.py [data_root]
Layout: <data_root>/<target>/reduced/*.fits ; targets that already have an output folder are skipped.
"""
import logging
import sys
from pathlib import Path

from variabledetection import TargetPipeline, discover_targets, load_config

logging.basicConfig(level=logging.INFO, format="%(message)s")

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\obsdata")
cfg = load_config(Path(__file__).with_name("config.toml"))

targets = discover_targets(ROOT, cfg, force=False)
print(f"{len(targets)} target(s) to process: {[t.name for t in targets]}")

summary = []
for target in targets:
    print(f"\n{'=' * 60}\n{target.name}\n{'=' * 60}")
    try:
        res = TargetPipeline(target, cfg).run()
        summary.append(f"{target.name}: {res.n_stars} stars, {len(res.flagged)} flagged {res.flagged}")
    except Exception as exc:  # keep going with the other targets
        summary.append(f"{target.name}: ERROR - {exc}")

print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
print("\n".join(summary))
