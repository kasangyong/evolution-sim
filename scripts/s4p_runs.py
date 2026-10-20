"""S4' (long-term survival): one million ticks for each seed, one subprocess per run, appended to
runs/<log>.log. Re-running skips seeds already logged. Usage: s4p_runs.py [log=NAME] [seed ...]
(default log s4p, seeds 1 3 6: a world without a bloom, one with bloom + snowball, and the
viewer's seed)."""
import os
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
name = next((a[4:] for a in args if a.startswith("log=")), "s4p")
seeds = [int(a) for a in args if not a.startswith("log=")] or [1, 3, 6]
root = Path(__file__).resolve().parents[1]
log = root / "runs" / f"{name}.log"
log.parent.mkdir(exist_ok=True)
done = log.read_text(encoding="utf-8") if log.exists() else ""
for seed in seeds:
    if f"seed {seed:>2} " in done:
        continue
    out = subprocess.run([sys.executable, str(root / "scripts" / "survival_run.py"), str(seed),
                          "1000000"], cwd=root, env={**os.environ, "PYTHONPATH": str(root)},
                         capture_output=True, text=True, check=False)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("seed")),
                f"seed {seed:>2} ERROR rc={out.returncode} {out.stderr.strip()[-200:]}")
    with log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
with log.open("a", encoding="utf-8") as f:
    f.write("DONE\n")
