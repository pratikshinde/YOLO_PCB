"""Run several experiments one after the other (unattended):  python run_queue.py E1_small E2_margin25 T0_base ...
For each: build datasets -> train 5 folds -> evaluate; at the end write results\\compare.md. A failed step is logged and the queue goes on."""
import subprocess
import sys
import time
from pathlib import Path

V3 = Path(__file__).parent
PY = str(V3.parent / ".venv" / "Scripts" / "python.exe")


def run(*args):
    t0 = time.time()
    r = subprocess.run([PY, *args], cwd=V3)
    print(f"### {' '.join(args)} -> exit {r.returncode} in {time.time() - t0:.0f}s", flush=True)
    return r.returncode


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    done = []
    for n in names:
        if run("build_datasets.py", n) == 0 and run("train_eval.py", n, "--workers", "4") == 0:
            done.append(n)
    if done:
        run("summarize.py", *(done if len(done) > 1 else done + done[:0]))
    print("QUEUE FINISHED:", done, flush=True)
