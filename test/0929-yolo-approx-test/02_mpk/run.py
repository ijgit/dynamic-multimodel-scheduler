"""MPK test: one GEMM DAG (dags/<tag>.json) as one MPK megakernel (sm_86, bf16, cutlass linear tasks).

  python 02_mpk/run.py --model n --batch 1 --res 640 --compile-only   # task graph + nvcc -> out/<tag>/, no launch
  python 02_mpk/run.py --model n --batch 1 --res 640                  # load, check, time, check again

Compile: once per DAG; out/<tag>/compile.json records the DAG's sha and a changed DAG is recompiled.
It uses the GPU only for MPK's initialization (allocations) and neither launches the kernel nor builds
the torch reference, so several compiles can share a GPU.
Run: load the compiled kernel (PersistentKernel.load_mpk_kernel), check the first launch against the
same DAG run with torch (01_torch's code path, bf16), warm up, time, check again. A launch runs the whole
DAG once (test_mode: one iteration, then the request retires); init_request_func() re-arms the request
before every later launch, outside the timed events.
Timing: CUDA events around launch_func (prepare kernel + worker and scheduler kernels, so the persistent
kernel's start-up is included, as a per-image invocation pays it).
The megakernel needs the GPU to itself: with another process doing CUDA work on the same GPU (even
allocations and copies), launches hang or end in an illegal memory access (reproduced 2026-09-29).
Runs therefore take a lock file per CUDA_VISIBLE_DEVICES, held from loading to finalize, and sweep.py
never runs two jobs on one GPU.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import mpk_common as C  # noqa: E402  (common/: CUDA 12 nvcc, c++17 pin, mpk_params)

C.setup()
import torch  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402

import dag as D  # noqa: E402
import mpk_exec as X  # noqa: E402
import runutil as U  # noqa: E402
from torch_exec import TorchDag  # noqa: E402


def make_pk(dag, out, weights, x, outputs):
    q = dag.summary()["mpk_queue_len"]
    if q != D.MPK_QUEUE:
        X.enable_worker_queue(q, os.path.join(out, "include_overlay"))
    pk = PersistentKernel(**C.mpk_params(D.MQ, use_cutlass_kernel=True))
    g = X.MPKDag(pk, dag, weights, x, outputs)
    return pk, dict(worker_queue_len=q, n_views=g.n_views)


def compiled(dag, out):
    try:
        cj = json.load(open(os.path.join(out, "compile.json")))
    except (OSError, ValueError):
        return False
    so = [n for n in os.listdir(out) if n.startswith("mpk_launcher_rank0") and n.endswith(".so")]
    return cj.get("dag_sha") == dag.sha() and bool(so) and os.path.exists(os.path.join(out, "task_graph_rank0.json"))


def compile_only(dag, out):
    cj = os.path.join(out, "compile.json")
    if os.path.exists(cj):
        os.remove(cj)
    x = D.make_input(dag)
    pk, info = make_pk(dag, out, D.make_weights(dag), x, D.alloc_outputs(dag, x))
    compile_s = C.compile_and_export(pk, out)
    X.check_cuda_error("MPK initialization after compile")
    pk.finalize()
    U.save_json(cj, dict(dag_sha=dag.sha(), dag=dag.summary(), compile_s=compile_s, gpu=U.gpu_info(), **info))
    return 0


def run(args, dag, out):
    lock = open(os.path.join(HERE, "out", f".gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    s = dag.summary()
    W, x = D.make_weights(dag), D.make_input(dag)
    outputs = D.alloc_outputs(dag, x)
    pk, info = make_pk(dag, out, W, x, outputs)
    pk.load_mpk_kernel(out)
    X.check_cuda_error("MPK initialization")
    res = dict(backend="mpk", dag=s, dag_sha=dag.sha(), gpu=U.gpu_info(), **info)
    ref = None if args.no_check else TorchDag(dag, W, x)
    stream = int(torch.cuda.current_stream().cuda_stream)

    def launch(first=False):
        if not first:
            pk.init_request_func()
        pk.launch_func(stream)

    def check(key):
        torch.cuda.synchronize()
        ref.forward()
        torch.cuda.synchronize()
        res[key] = U.compare(dag, outputs, ref.bufs)
        print(f"[check] {key}: MPK vs torch {U.fmt_check(res[key])}")
        return res[key]["passed"]

    t0 = time.perf_counter()
    launch(first=True)
    torch.cuda.synchronize()
    res["first_launch_wall_ms"] = (time.perf_counter() - t0) * 1e3
    ok = check("check_first") if ref is not None else True
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    iters = U.pick_iters(args, (time.perf_counter() - t0) * 1e3 / 3)

    def step(s, e):
        pk.init_request_func()  # a small init kernel and a stream sync, outside the events
        s.record()
        pk.launch_func(stream)
        e.record()

    tm = U.time_loop(step, iters, "mpk_run")
    res["timing"] = tm
    print(f"[run] mpk {dag.tag}: {tm['median_ms']:.3f} ms median over {iters} launches "
          f"({1e3 * args.batch / tm['median_ms']:.0f} img/s, {s['gflop'] / tm['median_ms']:.1f} TFLOPS executed; "
          f"wall {tm['wall_per_iter_ms']:.3f} ms/launch)")
    if ref is not None:
        for t in outputs.values():
            t.zero_()
        launch()
        ok = check("check") and ok
    U.save_json(args.result or os.path.join(out, "run.json"), res)
    pk.finalize()
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    U.add_common_args(ap)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--recompile", action="store_true")
    ap.add_argument("--max-gemms", type=int, default=0, help="debug: only the first N GEMMs (out/<tag>_first<N>/)")
    args = ap.parse_args()
    dag = D.load(args.model, args.batch, args.res)
    out = os.path.join(HERE, "out", dag.tag)
    if args.max_gemms:
        dag = D.truncate(dag, args.max_gemms)
        out += f"_first{args.max_gemms}"
    os.makedirs(out, exist_ok=True)
    s = dag.summary()
    print(f"[dag] {dag.tag}: {s['n_gemms']} GEMMs, {s['tasks']} tasks, {s['gflop']:.2f} GFLOP, sha {dag.sha()}")
    if args.compile_only:
        if compiled(dag, out) and not args.recompile:
            print(f"[compile] {out} is up to date")
            return 0
        return compile_only(dag, out)
    if args.recompile or not compiled(dag, out):
        subprocess.check_call([sys.executable, __file__, "--compile-only", "--recompile", "--model", args.model,
                               "--batch", str(args.batch), "--res", str(args.res), "--max-gemms", str(args.max_gemms)])
    return run(args, dag, out)


if __name__ == "__main__":
    sys.exit(main())
