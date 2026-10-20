"""S3 watch (24 h without a crash, process memory growth < 5%): every `every` seconds, log the live
viewer's tick, throughput, budget error, recording counters and the process's private memory to
runs/s3_monitor.log. Usage: s3_monitor.py [port] [every_s] [hours]. Run it detached; it stops after
`hours` or when the viewer disappears (logged as a crash)."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
every = int(sys.argv[2]) if len(sys.argv) > 2 else 600
hours = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
log = Path(__file__).resolve().parents[1] / "runs" / "s3_monitor.log"
log.parent.mkdir(exist_ok=True)


def powershell(cmd: str) -> str:
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                          text=True, timeout=60, check=False).stdout.strip()


def viewer_process() -> tuple[int, float, str] | None:
    """(pid, private MB, start time) of the process listening on the port."""
    out = powershell(
        f"$c = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue;"
        "if ($c) { $p = Get-Process -Id $c.OwningProcess;"
        " \"$($p.Id) $($p.PrivateMemorySize64) $($p.StartTime.ToString('s'))\" }")
    if not out:
        return None
    pid, mem, start = out.split()
    return int(pid), int(mem) / 2**20, start


def write(line: str):
    print(line, flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} watching port {port} every {every}s for {hours}h")
write("time                pid    started              private_MB  tick       tps   err      "
      "dropped errors")
t_end = time.time() + hours * 3600
while time.time() < t_end:
    proc = viewer_process()
    if proc is None:
        write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} VIEWER GONE (crash or stopped)")
        break
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/stats.json", timeout=20) as r:
            st = json.loads(r.read())
        stats = (f"{st['tick']:>10} {st['tps']:5.0f} {st['err']:.1e} {st.get('rec_dropped', 0):>7} "
                 f"{st.get('rec_errors', 0):>6}")
    except (OSError, ValueError) as e:
        stats = f"stats unavailable: {e}"
    pid, mem, start = proc
    write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {pid:>6} {start} {mem:10.0f}  {stats}")
    time.sleep(every)
