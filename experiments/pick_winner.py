"""Compare ReST chains and print a Pareto table to pick the generator to ship.

Reads experiments/rest/c*/history.json (metrics) + rest.log header (config). For each chain
shows the best round (by top100_reward) and its predicted profile, so the activity/selectivity/
Gram-balance tradeoff is visible at a glance. No wet-lab claims -- APEX/hemolysis predictions.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def config_of(log: Path) -> str:
    if not log.exists():
        return "?"
    head = log.read_text().splitlines()[:2]
    for line in head:
        m = re.search(r"hemo_lambda=(\S+).*gp_w=(\S+) mdr_w=(\S+)", line)
        if m:
            return f"lam={m.group(1)} gp={m.group(2)}"
    m2 = re.search(r"gp_w=(\S+) mdr_w=(\S+) hemo_lambda=(\S+)", "\n".join(head))
    if m2:
        return f"lam={m2.group(3)} gp={m2.group(1)}"
    return "?"


def main() -> int:
    rows = []
    for hist in sorted(glob.glob(str(REPO / "experiments/rest/c*/history.json"))):
        d = json.load(open(hist))
        chain = Path(hist).parent.name
        cfg = config_of(Path(hist).parent.parent / f"{chain}.log")
        best = max(d, key=lambda e: e.get("top100_reward", -9))
        r0 = d[0]
        rows.append((chain, cfg, best, r0, len(d) - 1))
    if not rows:
        print("no chains yet")
        return 0
    print(f"{'chain':<5} {'config':<16} {'rnd':>3} {'act%':>5} {'broad':>6} {'GN':>5} {'GP':>5} "
          f"{'MDR':>5} {'brd':>5} {'phemo':>6} {'novel':>6} {'div':>4} {'rew':>6}  (round0 broad->best)")
    for chain, cfg, b, r0, nround in sorted(rows):
        print(f"{chain:<5} {cfg:<16} {b.get('round','-'):>3} {b['pct_active']:>5.0f} "
              f"{b['SR_broad']:>6.1f} {b['SR_gn']:>5.1f} {b['SR_gp']:>5.1f} {b['SR_mdr']:>5.1f} "
              f"{b['mean_breadth']:>5.2f} {b.get('mean_phemo',0):>6.3f} {b['novel']:>6.3f} "
              f"{b.get('div150',0):>4} {b['top100_reward']:>6.3f}  "
              f"({r0['SR_broad']:.0f}->{b['SR_broad']:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
