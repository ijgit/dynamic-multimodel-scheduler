"""Helpers shared by 01_torch/run.py and 02_mpk/run.py."""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def add_common_args(ap):
    ap.add_argument("--model", default="n", help="arch/yolo26<model>.json")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--res", type=int, default=640, help="input height = width")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=0, help="timed iterations (0: enough for about --target-ms)")
    ap.add_argument("--target-ms", type=float, default=500.0, help="GPU time to cover when --iters is 0")
    ap.add_argument("--result", default=None, help="run.json path (default: <test>/out/<backend>/<tag>/run.json)")
    ap.add_argument("--no-check", action="store_true", help="skip the comparison with the reference")


def pick_iters(args, one_ms):
    if args.iters:
        return args.iters
    return int(min(1000, max(10, args.target_ms / max(one_ms, 1e-3))))


def time_loop(step, iters, nvtx_name):
    """Run step(start_event, end_event) iters times inside an NVTX range; per-iteration GPU times (ms)."""
    import torch

    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(nvtx_name)
    t0 = time.perf_counter()
    for s, e in ev:
        step(s, e)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    torch.cuda.nvtx.range_pop()
    ms = sorted(s.elapsed_time(e) for s, e in ev)
    return dict(iters=iters, wall_ms=wall, wall_per_iter_ms=wall / iters, median_ms=ms[len(ms) // 2], min_ms=ms[0],
                mean_ms=sum(ms) / len(ms), p90_ms=ms[int(0.9 * (len(ms) - 1))], max_ms=ms[-1])


def compare(dag, got, ref, tol=0.05):
    """Per GEMM: max |got - ref| / max |ref| over the valid rows of its output; passed if all < tol."""
    worst, rows = 0.0, []
    for gm in dag.gemms:
        n = dag.buffers[gm.name].pixels
        a, b = got[gm.name][:n].float(), ref[gm.name][:n].float()
        rel = ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()
        rows.append((gm.name, rel))
        worst = max(worst, rel)
    worst_gemm = max(rows, key=lambda r: r[1])[0]
    return dict(passed=worst < tol, tol=tol, worst_rel=worst, worst_gemm=worst_gemm, output_rel=dict(rows)[dag.output])


def fmt_check(c):
    return (f"{'PASSED' if c['passed'] else 'FAILED'} (worst rel {c['worst_rel']:.4f} at {c['worst_gemm']}, "
            f"output {c['output_rel']:.4f})")


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def gpu_info():
    import torch

    p = torch.cuda.get_device_properties(0)
    return dict(name=p.name, sms=p.multi_processor_count, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                torch=torch.__version__)
