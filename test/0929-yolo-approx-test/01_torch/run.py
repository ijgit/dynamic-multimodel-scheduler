"""PyTorch test: one GEMM DAG (dags/<tag>.json) with torch.mm / torch.addmm (cuBLAS, bf16).

  python 01_torch/run.py --model n --batch 1 --res 640            # eager: one cuBLAS call per GEMM
  python 01_torch/run.py --model n --batch 1 --res 640 --graph    # the same forward, captured once as a CUDA graph

02_mpk runs exactly this DAG (same file, weights and input). Timing: warm-up, then back-to-back forwards
inside the NVTX range "torch_run", CUDA events around each forward. Check: every GEMM output against the
same DAG in fp32; with --graph also graph replay == eager (bitwise).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import torch  # noqa: E402

import dag as D  # noqa: E402
import runutil as U  # noqa: E402
from torch_exec import TorchDag, reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    U.add_common_args(ap)
    ap.add_argument("--graph", action="store_true", help="capture one forward as a CUDA graph and replay it")
    args = ap.parse_args()
    dag = D.load(args.model, args.batch, args.res)
    backend = "torch_graph" if args.graph else "torch"
    s = dag.summary()
    res = dict(backend=backend, dag=s, dag_sha=dag.sha(), gpu=U.gpu_info())
    W = D.make_weights(dag)
    x = D.make_input(dag)
    net = TorchDag(dag, W, x)

    with torch.no_grad():
        for _ in range(max(args.warmup, 3)):
            net.forward()
        torch.cuda.synchronize()
        if args.graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                net.forward()
            run_once = g.replay
        else:
            run_once = net.forward
        s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s0.record()
        for _ in range(3):
            run_once()
        e0.record()
        torch.cuda.synchronize()
        iters = U.pick_iters(args, s0.elapsed_time(e0) / 3)

        def step(s, e):
            s.record()
            run_once()
            e.record()

        tm = U.time_loop(step, iters, "torch_run")
    res["timing"] = tm
    print(f"[run] {backend} {dag.tag}: {tm['median_ms']:.3f} ms median over {iters} forwards "
          f"({1e3 * args.batch / tm['median_ms']:.0f} img/s, {s['gflop'] / tm['median_ms']:.1f} TFLOPS executed; "
          f"wall {tm['wall_per_iter_ms']:.3f} ms/forward)")

    if not args.no_check:
        with torch.no_grad():
            if args.graph:
                got = {k: v.clone() for k, v in net.bufs.items()}
                net.forward()
                torch.cuda.synchronize()
                res["graph_vs_eager_equal"] = all(torch.equal(got[k], net.bufs[k]) for k in got)
            res["check"] = U.compare(dag, net.bufs, reference(dag, W, x).bufs)
        print(f"[check] bf16 vs fp32: {U.fmt_check(res['check'])}"
              + (f", graph == eager: {res['graph_vs_eager_equal']}" if args.graph else ""))
    U.save_json(args.result or os.path.join(HERE, "out", backend, dag.tag, "run.json"), res)
    ok = res.get("check", {}).get("passed", True) and res.get("graph_vs_eager_equal", True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
