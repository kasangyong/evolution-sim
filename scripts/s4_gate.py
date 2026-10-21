"""Resumable S4 survival gate: replicates × seeds, one subprocess per run, appended to
runs/<name>.log (argv[1], default s4_gate). Re-running skips (rep, seed) pairs already logged.
Further sect.key=val arguments are passed to every run; final genomes go to runs/genomes/<name>/."""
import subprocess
import sys
from pathlib import Path

REPS, SEEDS = 2, range(1, 11)
root = Path(__file__).resolve().parents[1]
log = root / "runs" / f"{sys.argv[1] if len(sys.argv) > 1 else 's4_gate'}.log"
log.parent.mkdir(exist_ok=True)
extra = sys.argv[2:]
genomes = root / "runs" / "genomes" / log.stem
genomes.mkdir(parents=True, exist_ok=True)
done = log.read_text(encoding="utf-8") if log.exists() else ""
for rep in range(1, REPS + 1):
    for seed in SEEDS:
        tag = f"rep {rep} seed {seed:>2} "
        if tag in done:
            continue
        out = subprocess.run([sys.executable, str(root / "scripts" / "survival_run.py"), str(seed), "100000",
                              *extra, f"save={genomes / f'rep{rep}_seed{seed}.npz'}"],
                             cwd=root, env={"PYTHONPATH": str(root), **__import__("os").environ},
                             capture_output=True, text=True, check=False)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("seed")),
                    f"seed {seed:>2} ERROR rc={out.returncode} {out.stderr.strip()[-200:]}")
        with log.open("a", encoding="utf-8") as f:
            f.write(f"rep {rep} {line}\n")
with log.open("a", encoding="utf-8") as f:
    f.write("DONE\n")
