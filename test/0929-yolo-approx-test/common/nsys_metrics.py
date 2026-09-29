#!/usr/bin/env python3
"""GPU resource usage of one test run with Nsight Systems (GPU Metrics sampling + CUDA kernel trace).

  python common/nsys_metrics.py run --out DIR --range torch_run -- $PY 01_torch/run.py --model n ...
  python nsys_metrics.py summarize DIR --range torch_run

`run` profiles the command once per metric set (one set per nsys session):
  ga10x       SM Active, SM Issue, Tensor Active, Compute Warps in Flight, Unallocated Warps, DRAM read/write
  ga10x-gfxt  CS Register Allocation, CS Shared Memory Allocated, L1/L2 throughput and hit rates,
              Warps Eligible, Active Thread Groups, SM (pipe) throughput, VRAM bandwidth
Both also trace CUDA kernels (grid, block, registers per thread, static + dynamic shared memory).
Values of GA10x metrics are % of peak (clocks in Hz). Samples are aggregated over the NVTX range the
test pushes around its timed loop ("range"), and over the samples that fall inside a kernel of that
range ("busy": gaps between kernels, e.g. MPK's per-launch re-arm, removed).

Output: DIR/<set>.sqlite (nsys report), DIR/<set>.log (command output), DIR/metrics.json:
  {"sets": {set: {"range": {metric: mean}, "busy": {...}, "p90": {...}}}, "kernels": [...], "kernel_summary": {...}}

Needs GPU performance counters (the container has CAP_SYS_ADMIN since 2026-09-28) and nsys 2023.1.2.
The GPU index for --gpu-metrics-device is CUDA_VISIBLE_DEVICES (nsys numbers GPUs like nvidia-smi).
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np

SETS = ("ga10x", "ga10x-gfxt")
# The metrics the tables report (the JSON keeps every metric of both sets).
KEY_METRICS = {
    "ga10x": ["SM Active", "SM Issue", "Tensor Active", "Compute Warps in Flight", "Unallocated Warps in Active SMs",
              "DRAM Read Throughput", "DRAM Write Throughput"],
    "ga10x-gfxt": ["CS Register Allocation", "CS Shared Memory Allocated (Sync)", "Compute Warps", "Warps Eligible",
                   "Active Thread Groups in SM", "SM Throughput", "SM Issue Active", "L1 Throughput", "L1 Hit Rate",
                   "L2 Throughput", "L2 Hit Rate", "L2 Hit Rate from L1", "VRAM Throughput"],
}
# RTX 3090 (GA102, sm_86) per-SM limits for theoretical occupancy
SM86 = dict(max_warps=48, max_blocks=16, regs=65536, reg_unit=256, smem=102400, smem_reserved=1024)


def gpu_index():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if not v.isdigit():
        raise SystemExit(f"set CUDA_VISIBLE_DEVICES to a GPU index (got {v!r})")
    return v


def theoretical_occupancy(block_threads, regs, smem_bytes, lim=SM86):
    """Warps per SM / 48 on sm_86 from the launch configuration (CUDA occupancy calculator rules)."""
    wpb = -(-block_threads // 32)
    by_warps = lim["max_warps"] // wpb
    regs_per_warp = -(-max(regs, 1) * 32 // lim["reg_unit"]) * lim["reg_unit"]
    by_regs = (lim["regs"] // regs_per_warp) // wpb
    by_smem = lim["smem"] // (smem_bytes + lim["smem_reserved"]) if smem_bytes > 0 else lim["max_blocks"]
    blocks = min(by_warps, by_regs, by_smem, lim["max_blocks"])
    limiter = min((by_warps, "warps"), (by_regs, "registers"), (by_smem, "shared memory"), (lim["max_blocks"], "blocks"))[1]
    return blocks * wpb / lim["max_warps"], blocks, limiter


def nvtx_range(c, name):
    rows = c.execute("select e.start, e.end from NVTX_EVENTS e left join StringIds s on e.textId = s.id "
                     "where (e.text = ? or s.value = ?) and e.end is not null order by e.start", (name, name)).fetchall()
    if not rows:
        raise SystemExit(f"no NVTX range named {name!r} in the report")
    return rows[-1]


def kernels_in(c, t0, t1):
    tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tabs:
        return []
    q = ("select k.start, k.end, s.value, k.gridX*k.gridY*k.gridZ, k.blockX*k.blockY*k.blockZ, k.registersPerThread, "
         "k.staticSharedMemory, k.dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s "
         "on k.shortName = s.id where k.end > ? and k.start < ? order by k.start")
    return c.execute(q, (t0, t1)).fetchall()


def busy_intervals(ks, t0, t1):
    out = []
    for s, e, *_ in ks:
        s, e = max(s, t0), min(e, t1)
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def samples(c, t0, t1):
    """(field names, [N, 3] array of timestamp, field index, value) of GpuMetrics samples in [t0, t1]."""
    src = [sid for sid, d in c.execute("select sourceId, data from GENERIC_EVENT_SOURCES")
           if json.loads(d).get("Name") == "GpuMetrics"]
    types = [tid for tid, sid in c.execute("select typeId, sourceId from GENERIC_EVENT_TYPES") if sid in src]
    if not types:
        return [], np.zeros((0, 3))
    names, idx, rows = [], {}, []
    q = ("select coalesce(timestamp, rawTimestamp) t, data from GENERIC_EVENTS where typeId in (%s) and "
         "coalesce(timestamp, rawTimestamp) between ? and ?" % ",".join(map(str, types)))
    for t, d in c.execute(q, (t0, t1)):
        for k, v in json.loads(d).items():
            if k not in idx:
                idx[k] = len(names)
                names.append(k)
            rows.append((t, idx[k], float(v)))
    return names, np.array(rows, dtype=np.float64).reshape(-1, 3)


def summarize_report(db, range_name):
    c = sqlite3.connect(db)
    t0, t1 = nvtx_range(c, range_name)
    ks = kernels_in(c, t0, t1)
    busy = busy_intervals(ks, t0, t1)
    names, data = samples(c, t0, t1)
    res = dict(range_ms=(t1 - t0) / 1e6, busy_ms=sum(e - s for s, e in busy) / 1e6, n_samples=0,
               range={}, busy={}, p90={})
    if len(data):
        inb = np.zeros(len(data), dtype=bool)
        ts = data[:, 0]
        starts = np.array([b[0] for b in busy])
        ends = np.array([b[1] for b in busy])
        if len(busy):
            j = np.searchsorted(starts, ts, side="right") - 1
            inb = (j >= 0) & (ts <= ends[np.clip(j, 0, None)])
        res["n_samples"] = int((data[:, 1] == 0).sum())
        for i, n in enumerate(names):
            m = data[:, 1] == i
            v = data[m, 2]
            res["range"][n] = float(v.mean())
            res["p90"][n] = float(np.percentile(v, 90))
            vb = data[m & inb, 2]
            res["busy"][n] = float(vb.mean()) if len(vb) else float("nan")
    return res, ks


def kernel_table(ks, iters=None):
    by = {}
    for s, e, name, grid, block, regs, ssm, dsm in ks:
        k = (name, grid, block, regs, ssm + dsm)
        d = by.setdefault(k, dict(name=name, grid=grid, block=block, regs=regs, smem=ssm + dsm, count=0, ns=0))
        d["count"] += 1
        d["ns"] += e - s
    tot = sum(d["ns"] for d in by.values()) or 1
    rows = []
    for d in sorted(by.values(), key=lambda d: -d["ns"]):
        occ, blocks, lim = theoretical_occupancy(d["block"], d["regs"], d["smem"])
        rows.append(dict(d, time_share=d["ns"] / tot, theo_occupancy=occ, blocks_per_sm=blocks, occ_limiter=lim))

    def tw(key):
        return sum(r[key] * r["ns"] for r in rows) / tot

    summ = dict(n_launches=sum(r["count"] for r in rows), n_distinct=len(rows), kernel_ms=tot / 1e6,
                tw_regs=tw("regs"), tw_smem_kb=tw("smem") / 1024, tw_block=tw("block"), tw_grid=tw("grid"),
                tw_theo_occupancy=tw("theo_occupancy"))
    if iters:
        summ["launches_per_iter"] = summ["n_launches"] / iters
    return rows, summ


def summarize(out_dir, range_name, iters=None):
    result = dict(range=range_name, sets={})
    for s in SETS:
        db = os.path.join(out_dir, f"{s}.sqlite")
        if not os.path.exists(db):
            continue
        res, ks = summarize_report(db, range_name)
        result["sets"][s] = res
        if s == SETS[0] or "kernels" not in result:
            result["kernels"], result["kernel_summary"] = kernel_table(ks, iters)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result


def print_summary(r):
    for s, res in r["sets"].items():
        print(f"  [{s}] range {res['range_ms']:.1f} ms, kernels busy {res['busy_ms']:.1f} ms, {res['n_samples']} samples")
        for k in KEY_METRICS[s]:
            if k in res["range"]:
                print(f"    {k:<36}{res['range'][k]:7.1f} (busy {res['busy'][k]:5.1f}, p90 {res['p90'][k]:5.1f})")
    ks = r.get("kernel_summary")
    if ks:
        print(f"  kernels: {ks['n_launches']} launches, {ks['n_distinct']} distinct; time-weighted regs/thread "
              f"{ks['tw_regs']:.0f}, smem/block {ks['tw_smem_kb']:.1f} KB, block {ks['tw_block']:.0f} threads, "
              f"theoretical occupancy {100 * ks['tw_theo_occupancy']:.0f}%")
        for k in r["kernels"][:5]:
            print(f"    {100 * k['time_share']:5.1f}%  x{k['count']:<6} grid {k['grid']:<6} block {k['block']:<4} "
                  f"regs {k['regs']:<4} smem {k['smem'] / 1024:5.1f} KB  occ {100 * k['theo_occupancy']:3.0f}% "
                  f"({k['occ_limiter']})  {k['name'][:60]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--range", required=True)
    r.add_argument("--sets", default=",".join(SETS))
    r.add_argument("--freq", type=int, default=20000, help="samples per second (gfxt needs >= 10000)")
    r.add_argument("command", nargs=argparse.REMAINDER)
    s = sub.add_parser("summarize")
    s.add_argument("out")
    s.add_argument("--range", required=True)
    a = ap.parse_args()
    if a.cmd == "summarize":
        print_summary(summarize(a.out, a.range))
        return 0
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    os.makedirs(a.out, exist_ok=True)
    rc = 0
    for mset in a.sets.split(","):
        rep = os.path.join(a.out, mset)
        nsys = ["nsys", "profile", "-o", rep, "--force-overwrite=true", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none",
                "--cpuctxsw=none", f"--gpu-metrics-device={gpu_index()}", f"--gpu-metrics-set={mset}",
                f"--gpu-metrics-frequency={a.freq}", "--export=sqlite"]
        with open(rep + ".log", "w") as log:
            log.write("+ " + " ".join(nsys + cmd) + "\n")
            log.flush()
            rc = subprocess.run(nsys + cmd, stdout=log, stderr=subprocess.STDOUT).returncode or rc
        if os.path.exists(rep + ".nsys-rep"):
            os.remove(rep + ".nsys-rep")  # the sqlite export has everything used here
    print_summary(summarize(a.out, a.range))
    return rc


if __name__ == "__main__":
    sys.exit(main())
