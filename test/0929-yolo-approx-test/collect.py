#!/usr/bin/env python3
"""Collect results/<backend>/<tag>/{run.json, nsys/metrics.json} into tables.

  python collect.py            # -> results/summary.csv (one row per backend x config), results/summary.md

Timing and checks come from the plain run (run.json), GPU metrics from the nsys run (nsys/metrics.json),
averaged over the NVTX range of the timed loop (% of peak). See README.md for how to read them.
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "common"))
import dag as D  # noqa: E402

BACKENDS = ["torch", "torch_graph", "mpk"]
# (column, nsys metric set, metric name)
METRICS = [
    ("sm_active", "ga10x", "SM Active"),
    ("sm_issue", "ga10x", "SM Issue"),
    ("tensor_active", "ga10x", "Tensor Active"),
    ("warps_in_flight", "ga10x", "Compute Warps in Flight"),
    ("unalloc_warps", "ga10x", "Unallocated Warps in Active SMs"),
    ("warps_eligible", "ga10x-gfxt", "Warps Eligible"),
    ("reg_alloc", "ga10x-gfxt", "CS Register Allocation"),
    ("smem_alloc", "ga10x-gfxt", "CS Shared Memory Allocated (Sync)"),
    ("l1_throughput", "ga10x-gfxt", "L1 Throughput"),
    ("l1_hit", "ga10x-gfxt", "L1 Hit Rate"),
    ("l2_throughput", "ga10x-gfxt", "L2 Throughput"),
    ("l2_hit", "ga10x-gfxt", "L2 Hit Rate"),
    ("dram_read", "ga10x", "DRAM Read Throughput"),
    ("dram_write", "ga10x", "DRAM Write Throughput"),
]
KERNEL = [("kernels_per_iter", "launches_per_iter"), ("regs_per_thread", "tw_regs"), ("smem_per_block_kb", "tw_smem_kb"),
          ("theo_occupancy", "tw_theo_occupancy")]
TITLES = {
    "median_ms": "Latency per forward (ms, median)", "img_s": "Throughput (images/s)",
    "tflops": "Executed TFLOPS (GEMM FLOPs of the DAG / latency)",
    "sm_active": "SM Active (% of SMs with a resident warp)", "sm_issue": "SM Issue (% of issue slots used)",
    "tensor_active": "Tensor Active (% of tensor-pipe cycles)",
    "warps_in_flight": "Compute Warps in Flight (% of 48 warps x 82 SMs)",
    "unalloc_warps": "Unallocated Warps in Active SMs (%)", "warps_eligible": "Warps Eligible (%)",
    "reg_alloc": "CS Register Allocation (% of the register file)",
    "smem_alloc": "CS Shared Memory Allocated (% of shared memory)", "l1_throughput": "L1 Throughput (% of peak)",
    "l1_hit": "L1 Hit Rate (%)", "l2_throughput": "L2 Throughput (% of peak)", "l2_hit": "L2 Hit Rate (%)",
    "dram_read": "DRAM Read Throughput (% of peak)", "dram_write": "DRAM Write Throughput (% of peak)",
    "kernels_per_iter": "Kernel launches per forward", "regs_per_thread": "Registers per thread (kernel-time weighted)",
    "smem_per_block_kb": "Shared memory per block (KB, kernel-time weighted)",
    "theo_occupancy": "Theoretical occupancy (%, kernel-time weighted, from registers/shared memory/block size)",
}


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def tags():
    order = {m: i for i, m in enumerate(D.MODELS)}
    found = set()
    for bk in BACKENDS:
        d = os.path.join(ROOT, "results", bk)
        if os.path.isdir(d):
            found |= set(os.listdir(d))

    def key(t):
        m, b, r = t.split("_")
        return order.get(m, len(order)), m, int(b[1:]), int(r[1:])

    configured = {f"{m}_b{b}_r{r}" for m in D.MODELS for b in D.BATCHES for r in D.RESOLUTIONS}
    return sorted(found & configured, key=key)  # results of other batches stay on disk but not in the tables


def rows():
    out = []
    for tag in tags():
        m, b, r = tag.split("_")
        b, r = int(b[1:]), int(r[1:])
        for bk in BACKENDS:
            d = os.path.join(ROOT, "results", bk, tag)
            run = load(os.path.join(d, "run.json"))
            met = load(os.path.join(d, "nsys", "metrics.json"))
            nrun = load(os.path.join(d, "nsys", "run.json"))
            if run is None and met is None:
                continue
            row = dict(backend=bk, model=m, batch=b, res=r, tag=tag)
            if run:
                p, t = run["dag"], run["timing"]
                chk = run.get("check") or {}
                row.update(median_ms=t["median_ms"], p90_ms=t["p90_ms"], img_s=1e3 * b / t["median_ms"],
                           tflops=p["gflop"] / t["median_ms"], wall_ms=t["wall_per_iter_ms"], iters=t["iters"],
                           gflop=p["gflop"], conv_gflop=p["real_gflop"], gemms=p["n_gemms"], tasks=p["tasks"],
                           check=chk.get("passed"), check_rel=chk.get("worst_rel"),
                           check_first=(run.get("check_first") or {}).get("passed"), dag_sha=run.get("dag_sha"))
            if met:
                for col, s, name in METRICS:
                    row[col] = met["sets"].get(s, {}).get("range", {}).get(name)
                ks = met.get("kernel_summary") or {}
                iters = (nrun or {}).get("timing", {}).get("iters")
                for col, key in KERNEL:
                    v = ks.get(key)
                    if key == "launches_per_iter" and iters:
                        v = ks.get("n_launches", 0) / iters
                    row[col] = v * 100 if key == "tw_theo_occupancy" and v is not None else v
            out.append(row)
    return out


def fmt(v, col):
    if v is None:
        return "-"
    if col in ("img_s", "kernels_per_iter"):
        return f"{v:.0f}"
    if col in ("median_ms", "p90_ms", "wall_ms"):
        return f"{v:.3f}" if v < 10 else f"{v:.2f}"
    if col == "ratio":
        return f"{v:.2f}"
    return f"{v:.1f}"


def main():
    rs = rows()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    cols = ["backend", "model", "batch", "res", "median_ms", "p90_ms", "img_s", "tflops", "wall_ms", "iters", "gflop",
            "conv_gflop", "gemms", "tasks", "check", "check_rel", "check_first", "dag_sha"] + \
        [c for c, _, _ in METRICS] + [c for c, _ in KERNEL]
    with open(os.path.join(ROOT, "results", "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rs)
    idx = {(r["backend"], r["tag"]): r for r in rs}
    ts = [t for t in tags() if any((bk, t) in idx for bk in BACKENDS)]
    bks = [bk for bk in BACKENDS if any((bk, t) in idx for t in ts)]
    md = ["# YOLO26 backbone as a GEMM-only DAG: torch vs MPK", "",
          "Generated by `collect.py` from `results/`. Latency: CUDA events, median over the timed loop. "
          "Metrics: nsys GPU Metrics, mean over the timed loop's NVTX range (% of peak).",
          "Backends: " + ", ".join(f"`{b}`" for b in bks) + " (see README.md).", ""]

    def table(title, header, cells):
        md.extend([f"## {title}", "", "| model | batch | res | " + " | ".join(header) + " |",
                   "|---|---|---|" + "---|" * len(header)])
        for t in ts:
            m, b, r = t.split("_")
            md.append(f"| {m} | {b[1:]} | {r[1:]} | " + " | ".join(cells(t)) + " |")
        md.append("")

    def get(bk, t, col):
        return idx.get((bk, t), {}).get(col)

    def ratio(t, num, den):
        a, b = get(num, t, "median_ms"), get(den, t, "median_ms")
        return fmt(a / b, "ratio") if a and b else "-"

    if "mpk" in bks:
        others = [bk for bk in bks if bk != "mpk"]
        table("MPK latency / torch latency (< 1: MPK faster)", [f"mpk / {bk}" for bk in others],
              lambda t: [ratio(t, "mpk", bk) for bk in others])

    def check_cell(bk, t):
        r = idx.get((bk, t))
        if r is None or r.get("check") is None:
            return "-"
        first = "" if r.get("check_first") in (None, True) else " (first launch FAILED)"
        return f"{'ok' if r['check'] else 'FAILED'} {r['check_rel']:.3f}{first}"

    table("Checks (torch: bf16 vs fp32 DAG; mpk: vs torch bf16; worst per-GEMM max|diff| / max|ref|)", bks,
          lambda t: [check_cell(bk, t) for bk in bks])
    for col in ["median_ms", "img_s", "tflops", "wall_ms"] + [c for c, _, _ in METRICS] + [c for c, _ in KERNEL]:
        title = TITLES.get(col, "Host wall time per forward (ms)")
        table(title, bks, lambda t, col=col: [fmt(get(bk, t, col), col) for bk in bks])
    with open(os.path.join(ROOT, "results", "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"{len(rs)} runs -> results/summary.csv, results/summary.md")


if __name__ == "__main__":
    main()
