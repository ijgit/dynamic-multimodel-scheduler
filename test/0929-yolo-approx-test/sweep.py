#!/usr/bin/env python3
"""Run every GEMM DAG with torch (eager, CUDA graph) and MPK, and collect GPU resource usage with nsys.

  source env.sh
  setsid nohup $PY sweep.py --gpus 6 > results/logs/sweep.nohup.out 2>&1 < /dev/null &
  python sweep.py --models n --batches 1 --backends torch,mpk
  python sweep.py --no-nsys                       # timing and checks only
  python collect.py                               # tables -> results/summary.md, results/summary.csv

Backends (one directory each under results/):
  torch        01_torch/run.py           eager, one cuBLAS call per GEMM
  torch_graph  01_torch/run.py --graph   the same forward replayed as one CUDA graph
  mpk          02_mpk/run.py             one MPK megakernel launch per forward
Steps: (0) common/dag.py writes the DAGs (unchanged files are left alone); (1) every MPK kernel whose
DAG changed is compiled (nvcc, --compile-jobs at once, spread over the GPUs; compiles do not launch);
then, one worker per GPU, large configurations first, one job per GPU at a time:
  (2) plain runs with checks, every backend -> results/<backend>/<tag>/run.json
  (3) nsys for torch and torch_graph        -> results/<backend>/<tag>/nsys/metrics.json
  (4) nsys for mpk, last: an MPK launch under nsys once hung and wedged the GPU (2026-09-29), so it runs
      after everything else is in, and only for configurations whose plain MPK run passed.
There is no timeout: a hung job stops the sweep there (see sh()). The nsys sqlite reports (up to about
1 GB each) are deleted once summarized unless --keep-reports. Logs: results/logs/ (sweep.log is this
script's own log).
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "common"))
import dag as D  # noqa: E402

PY = os.environ.get("PY", sys.executable)
BACKENDS = {
    "torch": (["01_torch/run.py"], "torch_run"),
    "torch_graph": (["01_torch/run.py", "--graph"], "torch_run"),
    "mpk": (["02_mpk/run.py"], "mpk_run"),
}
LOGDIR = os.path.join(ROOT, "results", "logs")
lock = threading.Lock()


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with lock:
        print(line, flush=True)
        with open(os.path.join(LOGDIR, "sweep.log"), "a") as f:
            f.write(line + "\n")


def sh(cmd, logfile, gpu):
    """No timeout: a hung MPK kernel is not cleaned up by killing its process tree, and starting the next job on
    that GPU wedges it (2026-09-29). A hang stops the sweep; look at it and decide by hand."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(logfile, "a") as f:
        f.write(f"\n+ [{time.strftime('%H:%M:%S')}] CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode


def cfg_args(m, b, r):
    return ["--model", m, "--batch", str(b), "--res", str(r)]


def compiled(m, b, r):
    try:
        cj = json.load(open(os.path.join(ROOT, "02_mpk", "out", f"{m}_b{b}_r{r}", "compile.json")))
    except (OSError, ValueError):
        return False
    return cj.get("dag_sha") == D.load(m, b, r).sha()


def compile_all(cfgs, gpus, jobs):
    todo = [c for c in cfgs if not compiled(*c)]
    log(f"[compile] {len(todo)} MPK kernels to build, {jobs} at a time on GPUs {gpus}")
    q = queue.Queue()
    for i, c in enumerate(todo):
        q.put((i, c))

    def worker():
        while True:
            try:
                i, (m, b, r) = q.get_nowait()
            except queue.Empty:
                return
            tag = f"{m}_b{b}_r{r}"
            t0 = time.time()
            rc = sh([PY, "02_mpk/run.py", "--compile-only"] + cfg_args(m, b, r),
                    os.path.join(LOGDIR, f"compile_{tag}.log"), gpus[i % len(gpus)])
            log(f"[compile] {tag}: rc={rc} {time.time() - t0:.0f} s")

    ts = [threading.Thread(target=worker) for _ in range(jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    missing = [f"{m}_b{b}_r{r}" for m, b, r in todo if not compiled(m, b, r)]
    if missing:
        log(f"[compile] FAILED: {' '.join(missing)}")


def plain_job(backend, m, b, r, gpu, a):
    tag = f"{m}_b{b}_r{r}"
    script, _ = BACKENDS[backend]
    d = os.path.join(ROOT, "results", backend, tag)
    if backend == "mpk" and not compiled(m, b, r):
        log(f"[skip] mpk {tag}: no compiled kernel")
        return
    if a.skip_existing and os.path.exists(os.path.join(d, "run.json")):
        return
    base = [PY] + script + cfg_args(m, b, r) + ["--target-ms", str(a.target_ms)]
    t0 = time.time()
    for attempt in range(1, a.attempts + 1):
        rc = sh(base + ["--result", os.path.join(d, "run.json")], os.path.join(LOGDIR, f"{backend}_{tag}.log"), gpu)
        if rc == 0:
            break
        log(f"[run] {backend} {tag} gpu {gpu}: attempt {attempt} rc={rc}")
    log(f"[run] {backend:<11} {tag:<12} gpu {gpu}: rc={rc} {time.time() - t0:.0f} s")


def nsys_job(backend, m, b, r, gpu, a):
    tag = f"{m}_b{b}_r{r}"
    script, rng = BACKENDS[backend]
    d = os.path.join(ROOT, "results", backend, tag)
    nd = os.path.join(d, "nsys")
    if a.skip_existing and os.path.exists(os.path.join(nd, "metrics.json")):
        return
    try:
        passed = (json.load(open(os.path.join(d, "run.json"))).get("check") or {}).get("passed")
    except (OSError, ValueError):
        passed = None
    if backend == "mpk" and not passed:
        log(f"[skip] nsys mpk {tag}: the plain run did not pass")
        return
    base = [PY] + script + cfg_args(m, b, r) + ["--target-ms", str(a.target_ms)]
    t0 = time.time()
    rc = sh([PY, "common/nsys_metrics.py", "run", "--out", nd, "--range", rng, "--freq", str(a.freq), "--"] + base
            + ["--no-check", "--result", os.path.join(nd, "run.json")], os.path.join(LOGDIR, f"{backend}_{tag}.log"), gpu)
    if not a.keep_reports:
        for s in ("ga10x", "ga10x-gfxt"):
            p = os.path.join(nd, f"{s}.sqlite")
            if os.path.exists(p):
                os.remove(p)
    log(f"[nsys] {backend:<11} {tag:<12} gpu {gpu}: rc={rc} {time.time() - t0:.0f} s")


def run_phase(name, fn, jobs, gpus, a):
    """Jobs over one worker per GPU; returns when all are done."""
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    log(f"[{name}] {len(jobs)} jobs")

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            fn(*j, gpu, a)

    ts = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(D.MODELS))
    ap.add_argument("--batches", default=",".join(map(str, D.BATCHES)))
    ap.add_argument("--res", default=",".join(map(str, D.RESOLUTIONS)))
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--gpus", default=os.environ.get("MPK_GPU", "6"), help="comma-separated GPU indices, one worker each")
    ap.add_argument("--compile-jobs", type=int, default=8)
    ap.add_argument("--target-ms", type=float, default=500.0, help="GPU time per timed loop")
    ap.add_argument("--freq", type=int, default=20000, help="nsys GPU metrics sampling rate (Hz)")
    ap.add_argument("--attempts", type=int, default=2, help="plain runs per job until one passes (failures are logged)")
    ap.add_argument("--no-nsys", action="store_true")
    ap.add_argument("--keep-reports", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    os.makedirs(LOGDIR, exist_ok=True)
    gpus = [g.strip() for g in a.gpus.split(",")]
    backends = a.backends.split(",")
    assert all(bk in BACKENDS for bk in backends), backends
    subprocess.check_call([PY, "common/dag.py", "--models", a.models, "--batches", a.batches, "--res", a.res], cwd=ROOT,
                          stdout=subprocess.DEVNULL)
    cfgs = [(m, int(b), int(r)) for m in a.models.split(",") for b in a.batches.split(",") for r in a.res.split(",")]
    log(f"[sweep] {len(cfgs)} configurations x {backends} on GPUs {gpus}, nsys {'off' if a.no_nsys else 'on'}")
    if "mpk" in backends:
        compile_all(cfgs, gpus, a.compile_jobs)
    # large configurations first so that the GPUs finish at about the same time
    cfgs.sort(key=lambda c: -D.load(*c).summary()["gflop"])
    run_phase("run", plain_job, [(bk, *c) for c in cfgs for bk in backends], gpus, a)
    if not a.no_nsys:
        run_phase("nsys torch", nsys_job, [(bk, *c) for c in cfgs for bk in backends if bk != "mpk"], gpus, a)
        if "mpk" in backends:
            run_phase("nsys mpk", nsys_job, [("mpk", *c) for c in cfgs], gpus, a)
    log("[done] python collect.py for the tables")


if __name__ == "__main__":
    main()
