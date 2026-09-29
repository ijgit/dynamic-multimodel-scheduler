"""GEMM-only DAG of the YOLO26 backbone for one (model, batch, resolution).

  python common/dag.py                         # every configuration -> dags/<tag>.json, dags/summary.md
  python common/dag.py --show n 1 640          # per-GEMM table of one configuration

Input: arch/yolo26<model>.json, the conv DAG of the real network (arch/gen_arch.py).
Output: a DAG whose every node is one GEMM   out[M, N] = A[M, K] @ W[N, K]^T  (+ res[M, N]).
Each GEMM writes its own buffer, a row-major activation [rows, C] (one row per pixel, NHWC). A GEMM
reads the first M*K elements of its A buffer as [M, K] and writes (and adds) whole buffers as [M, N].
Nothing else runs on the GPU: no im2col, pooling, normalization or activation.

Conv -> GEMM. Pixel grouping: [rows, C] read as [rows/g, g*C] puts g consecutive pixels in one GEMM row,
and a dense W[g*Cout, g*Cin] makes each output pixel depend on the g input pixels of its group.
    conv            GEMM (M, N, K)                  FLOPs vs the conv
    1x1             (P/g, g*Cout, g*Cin)            g   (g = 1 unless the channels are too few for MPK)
    3x3, stride 1   (P/g, g*Cout, g*Cin), g = 8     8/9 (receptive field: 8 consecutive pixels, not 3x3)
    3x3, stride 2   (P/g, g*Cout, g*f*Cin), g*f = 8 8/9 (first f*P input rows, space-to-depth; f = 4 or 2)
    stem 3x3 s2     (P/g, g*Cout, g*32)             1   (graph input = host im2col, 27 -> 32 columns)
    concat -> 1x1   one GEMM per part, chained through the residual: out_j = part_j @ W_j^T + out_prev
    shortcut add    residual of the conv's (last) GEMM
P = rows of the output level. g is the smallest value with K % 128 == 0 and N % 64 == 0 (MPK linear
task, cutlass on sm_86: 16 x 64 output tile per task; K = 64 gives wrong results), for 3x3 its lcm with 8.
The parts of one concat share g (one output layout). Rows of a level = its pixels rounded up to
16 * lcm(g of the GEMMs writing that level); f of a stride-2 GEMM depends on those rows (fixed point).
Parts are summed latest-produced first: MPK rejects a GEMM that both forks and feeds a join
(annotated_graph.cc); in this order the extra edge is implied by a longer path and MPK drops it.
"""
import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from math import gcd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH_DIR = os.path.join(ROOT, "arch")
DAG_DIR = os.path.join(ROOT, "dags")
MODELS = ("n", "m", "l")
BATCHES = (1, 8)
RESOLUTIONS = (240, 320, 640)
KQ, NQ, MQ = 128, 64, 16  # MPK linear task: K granularity, output columns per task, output rows per task
STEM_K = 32               # stem im2col columns: 3*3*3 = 27, padded
RF = 8                    # pixels per group of a 3x3 conv (the real receptive field is 9)
MPK_WORKERS, MPK_QUEUE = 64, 8192  # a launch queues all its tasks at start: 64 workers x 8192 by default
INPUT = "image_cols"


def lcm(*xs):
    out = 1
    for x in xs:
        out = out * x // gcd(out, x)
    return out


def g_k(c):  # smallest g with g*c % KQ == 0
    return KQ // gcd(c, KQ)


def g_n(c):  # smallest g with g*c % NQ == 0
    return NQ // gcd(c, NQ)


def roundup(x, m):
    return (x + m - 1) // m * m


def level_hw(res, level):
    h = res
    for _ in range(level):
        h = (h + 1) // 2  # 3x3, stride 2, padding 1
    return h


@dataclass
class Buffer:
    name: str
    level: int       # 1..5 = P1..P5 (stride 2..32)
    C: int
    rows: int = 0    # the buffer is [rows, C]
    pixels: int = 0  # valid rows: batch * h * w of the level


@dataclass
class Gemm:
    name: str          # also the name of the buffer it writes
    src: str           # conv in arch/yolo26<model>.json
    kind: str          # stem | 1x1 | 3x3 | 3x3s2 | cat (one part of a concat -> 1x1)
    a: str             # A operand: the first M*K elements of this buffer
    res: str = None    # residual buffer, [M, N]
    chain: str = ""    # the parts of one concat share it (and g)
    cin: int = 0       # channels per pixel of A
    cout: int = 0
    f: int = 1         # input rows per output row (stride 2)
    g: int = 1         # pixels per GEMM row
    M: int = 0
    N: int = 0
    K: int = 0
    real_flops: float = 0.0  # FLOPs of the conv (or concat part) it stands for

    @property
    def tasks(self):  # MPK tasks (16 x 64 output tiles)
        return (self.M // MQ) * (self.N // NQ)

    @property
    def flops(self):
        return 2.0 * self.M * self.N * self.K


@dataclass
class Dag:
    model: str
    batch: int
    res: int
    levels: dict   # level -> dict(hw, pixels, rows, align)
    buffers: dict  # name -> Buffer (the input and one per GEMM)
    gemms: list    # execution order
    input: str
    output: str

    @property
    def tag(self):
        return f"{self.model}_b{self.batch}_r{self.res}"

    def layout(self, name):
        """[M, N] in which a buffer is written: its producer's GEMM output (the input: [rows, C])."""
        p = self._producer.get(name)
        b = self.buffers[name]
        return (p.M, p.N) if p is not None else (b.rows, b.C)

    @property
    def _producer(self):
        return {gm.name: gm for gm in self.gemms}

    def summary(self):
        tasks = sum(gm.tasks for gm in self.gemms)
        fl = sum(gm.flops for gm in self.gemms)
        real = sum(gm.real_flops for gm in self.gemms)
        pix = sum(v["pixels"] for v in self.levels.values())
        rows = sum(v["rows"] for v in self.levels.values())
        q = MPK_QUEUE
        while q * MPK_WORKERS < tasks + 2:
            q *= 2
        return dict(tag=self.tag, model=self.model, batch=self.batch, res=self.res, n_gemms=len(self.gemms),
                    n_residual=sum(gm.res is not None for gm in self.gemms), tasks=tasks, gflop=fl / 1e9,
                    real_gflop=real / 1e9, flop_ratio=fl / real,
                    weight_mb=sum(2 * gm.N * gm.K for gm in self.gemms) / 1e6,
                    act_mb=sum(2 * b.rows * b.C for b in self.buffers.values()) / 1e6, row_padding=rows / pix - 1,
                    n_shapes=len({(gm.K, gm.N, gm.res is not None) for gm in self.gemms}), mpk_queue_len=q)

    # ---- serialization: dags/<tag>.json, one buffer / GEMM per line

    def body(self):
        return dict(model=self.model, batch=self.batch, res=self.res, input=self.input, output=self.output,
                    levels={str(k): v for k, v in self.levels.items()},
                    buffers=[asdict(b) for b in self.buffers.values()], gemms=[asdict(gm) for gm in self.gemms])

    def sha(self):
        return hashlib.sha1(json.dumps(self.body(), sort_keys=True).encode()).hexdigest()[:12]

    def to_json(self):
        body = self.body()
        head = dict(tag=self.tag, sha=self.sha(), summary=self.summary(),
                    constants=dict(KQ=KQ, NQ=NQ, MQ=MQ, STEM_K=STEM_K, RF=RF),
                    **{k: body[k] for k in ("model", "batch", "res", "input", "output", "levels")})
        out = "{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in head.items())
        for k in ("buffers", "gemms"):
            out += f',\n {json.dumps(k)}: [\n' + ",\n".join("  " + json.dumps(x) for x in body[k]) + "\n ]"
        return out + "\n}\n"

    @staticmethod
    def from_json(text):
        d = json.loads(text)
        bufs = {b["name"]: Buffer(**b) for b in d["buffers"]}
        names = {f.name for f in fields(Gemm)}
        gemms = [Gemm(**{k: v for k, v in gm.items() if k in names}) for gm in d["gemms"]]
        dag = Dag(d["model"], d["batch"], d["res"], {int(k): v for k, v in d["levels"].items()}, bufs, gemms,
                  d["input"], d["output"])
        assert dag.sha() == d["sha"], f"{dag.tag}: sha mismatch (file edited?)"
        return dag


def load_arch(model):
    path = os.path.join(ARCH_DIR, f"yolo26{model}.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run arch/gen_arch.py --models {model}")
    return json.load(open(path))


def logical(arch):
    """Conv DAG -> GEMMs in execution order with kinds, operands and channels (no shapes yet)."""
    image = arch["input"]["name"]
    bufs = {INPUT: Buffer(INPUT, 1, STEM_K)}
    gemms, pos = [], {INPUT: -1}  # pos: emission index of a buffer's producer
    tensor = {image: INPUT}       # conv output / image -> buffer holding it
    level = {image: 0}

    def emit(gm, lv):
        bufs[gm.name] = Buffer(gm.name, lv, gm.cout)
        pos[gm.name] = len(gemms)
        gemms.append(gm)
        return gm.name

    for n in arch["convs"]:
        lvs = {level[i] for i in n["in"]}
        assert len(lvs) == 1, n["name"]
        lv = lvs.pop() + (1 if n["s"] == 2 else 0)
        level[n["name"]] = lv
        res = tensor[n["res"]] if "res" in n else None
        cout = n["cout"]
        if n["in"] == [image]:
            assert (n["k"], n["s"], n["cin"]) == (3, 2, 3), n["name"]
            tensor[n["name"]] = emit(Gemm(n["name"], n["name"], "stem", INPUT, res, n["name"], STEM_K, cout), lv)
        elif len(n["in"]) > 1:  # concat -> 1x1: partial GEMMs chained through the residual
            assert n["k"] == 1 and n["s"] == 1, n["name"]
            parts = [tensor[i] for i in n["in"]]
            prev = res
            for j in sorted(range(len(parts)), key=lambda j: -pos[parts[j]]):
                prev = emit(Gemm(f"{n['name']}.p{j}", n["name"], "cat", parts[j], prev, n["name"], bufs[parts[j]].C,
                                 cout), lv)
            tensor[n["name"]] = prev
        else:
            assert n["k"] in (1, 3) and (n["k"] == 3 or n["s"] == 1), n["name"]
            kind = "1x1" if n["k"] == 1 else ("3x3" if n["s"] == 1 else "3x3s2")
            a = tensor[n["in"][0]]
            assert bufs[a].C == n["cin"], n["name"]
            tensor[n["name"]] = emit(Gemm(n["name"], n["name"], kind, a, res, n["name"], n["cin"], cout), lv)
    return gemms, bufs, tensor[arch["output"]]


def build(arch, batch, res):
    """Shapes of the GEMM DAG for one (batch, resolution)."""
    gemms, bufs, output = logical(arch)
    levels = {l: dict(hw=level_hw(res, l), pixels=batch * level_hw(res, l) ** 2) for l in range(1, 6)}
    out_level = {gm.name: bufs[gm.name].level for gm in gemms}
    f_of = {gm.name: 4 for gm in gemms if gm.kind == "3x3s2"}
    for _ in range(8):  # fixed point: g depends on f, rows depend on g, f depends on rows
        g_of = {}
        for gm in gemms:
            if gm.kind == "3x3":
                g = lcm(RF, g_k(gm.cin), g_n(gm.cout))
            elif gm.kind == "3x3s2":
                f = f_of[gm.name]
                g = lcm(RF // f, g_k(f * gm.cin), g_n(gm.cout))
            else:  # stem, 1x1, cat
                g = lcm(g_k(gm.cin), g_n(gm.cout))
            g_of[gm.name] = g
        for chain in {gm.chain for gm in gemms}:
            members = [gm.name for gm in gemms if gm.chain == chain]
            g = lcm(*(g_of[n] for n in members))
            for n in members:
                g_of[n] = g
        for l, v in levels.items():
            v["align"] = MQ * lcm(*(g_of[gm.name] for gm in gemms if out_level[gm.name] == l))
            v["rows"] = roundup(v["pixels"], v["align"])
        new_f = {}
        for gm in gemms:
            if gm.kind == "3x3s2":
                ratio = levels[bufs[gm.a].level]["rows"] // levels[out_level[gm.name]]["rows"]
                new_f[gm.name] = 4 if ratio >= 4 else (2 if ratio >= 2 else 1)
        if new_f == f_of:
            break
        f_of = new_f
    else:
        raise RuntimeError("row planning did not converge")
    for b in bufs.values():
        b.rows, b.pixels = levels[b.level]["rows"], levels[b.level]["pixels"]
    for gm in gemms:
        gm.f, gm.g = f_of.get(gm.name, 1), g_of[gm.name]
        rows = levels[out_level[gm.name]]["rows"]
        gm.M, gm.N, gm.K = rows // gm.g, gm.g * gm.cout, gm.g * gm.f * gm.cin
        real_k = {"stem": 27, "3x3": 9 * gm.cin, "3x3s2": 9 * gm.cin}.get(gm.kind, gm.cin)
        gm.real_flops = 2.0 * levels[out_level[gm.name]]["pixels"] * gm.cout * real_k
    dag = Dag(arch["model"], batch, res, levels, bufs, gemms, INPUT, output)
    validate(dag)
    return dag


def validate(dag):
    """Shape rules both executors rely on."""
    for gm in dag.gemms:
        a, o = dag.buffers[gm.a], dag.buffers[gm.name]
        assert gm.M % MQ == 0 and gm.N % NQ == 0 and gm.K % KQ == 0, (gm.name, gm.M, gm.N, gm.K)
        assert gm.M * gm.N == o.rows * o.C, gm.name                     # writes the whole output buffer
        assert gm.K == gm.g * gm.f * a.C and gm.M * gm.K <= a.rows * a.C, gm.name  # reads a prefix of A
        assert (gm.M * gm.K) % dag.layout(gm.a)[1] == 0, gm.name         # ... made of whole rows of A's layout
        if gm.res is not None:
            r = dag.buffers[gm.res]
            assert r.rows * r.C == gm.M * gm.N and r.level == o.level, gm.name
    names = [gm.name for gm in dag.gemms]
    assert len(set(names)) == len(names) and dag.output in dag.buffers
    seen = {dag.input}
    for gm in dag.gemms:  # topological order
        assert gm.a in seen and (gm.res is None or gm.res in seen), gm.name
        seen.add(gm.name)


def truncate(dag, n):
    """The first n GEMMs of a DAG (still a valid DAG; its output is GEMM n-1). For debugging."""
    gemms = [Gemm(**asdict(gm)) for gm in dag.gemms[:n]]
    keep = {dag.input} | {gm.name for gm in gemms}
    bufs = {k: Buffer(**asdict(b)) for k, b in dag.buffers.items() if k in keep}
    return Dag(dag.model, dag.batch, dag.res, dag.levels, bufs, gemms, dag.input, gemms[-1].name)


def path(model, batch, res):
    return os.path.join(DAG_DIR, f"{model}_b{batch}_r{res}.json")


def load(model, batch, res):
    p = path(model, batch, res)
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing: run python common/dag.py")
    return Dag.from_json(open(p).read())


# ----------------------------------------------------------------------------------------------
# Tensors shared by both executors (torch is imported only here)


def make_weights(dag, device="cuda", dtype=None, seed=0):
    """W[N, K] per GEMM, N(0, 1/K) so activations keep their scale."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    return {gm.name: (torch.randn(gm.N, gm.K, generator=gen) * gm.K ** -0.5).to(device, dtype or torch.bfloat16)
            for gm in dag.gemms}


def make_input(dag, device="cuda", dtype=None, seed=1):
    """[rows(P1), 32]: random im2col values in the valid rows and first 27 columns, zeros elsewhere."""
    import torch

    b = dag.buffers[dag.input]
    x = torch.zeros(b.rows, b.C)
    x[: b.pixels, :27] = torch.randn(b.pixels, 27, generator=torch.Generator().manual_seed(seed))
    return x.to(device, dtype or torch.bfloat16)


def alloc_outputs(dag, like):
    """One zeroed [rows, C] tensor per GEMM output."""
    import torch

    return {gm.name: torch.zeros(dag.buffers[gm.name].rows, gm.cout, dtype=like.dtype, device=like.device)
            for gm in dag.gemms}


# ----------------------------------------------------------------------------------------------


def table(dag):
    s = dag.summary()
    out = [f"YOLO26{dag.model} backbone as {s['n_gemms']} GEMMs, batch {dag.batch}, {dag.res}x{dag.res}: "
           f"{s['tasks']} MPK tasks, {s['gflop']:.2f} GFLOP executed vs {s['real_gflop']:.2f} for the convs "
           f"({s['flop_ratio']:.2f}x), weights {s['weight_mb']:.1f} MB, activations {s['act_mb']:.1f} MB, "
           f"row padding {100 * s['row_padding']:.1f}%, sha {dag.sha()}"]
    for l, v in dag.levels.items():
        out.append(f"  P{l}: {v['hw']}x{v['hw']} x{dag.batch} = {v['pixels']} px -> {v['rows']} rows "
                   f"(multiple of {v['align']})")
    out.append(f"  {'GEMM':<24}{'kind':<6}{'A':<22}{'f':>2}{'g':>3}{'M':>8}{'N':>6}{'K':>6}{'tasks':>8}"
               f"{'GFLOP':>8}{'conv':>7}  residual")
    for gm in dag.gemms:
        out.append(f"  {gm.name:<24}{gm.kind:<6}{gm.a:<22}{gm.f:>2}{gm.g:>3}{gm.M:>8}{gm.N:>6}{gm.K:>6}{gm.tasks:>8}"
                   f"{gm.flops / 1e9:>8.3f}{gm.real_flops / 1e9:>7.3f}  {gm.res or ''}")
    return "\n".join(out)


SUMMARY_COLS = [("model", "model", "{}"), ("batch", "batch", "{}"), ("res", "res", "{}"), ("GEMMs", "n_gemms", "{}"),
                ("residual", "n_residual", "{}"), ("MPK tasks", "tasks", "{}"), ("GFLOP", "gflop", "{:.2f}"),
                ("conv GFLOP", "real_gflop", "{:.2f}"), ("ratio", "flop_ratio", "{:.2f}"),
                ("weights MB", "weight_mb", "{:.1f}"), ("act MB", "act_mb", "{:.1f}"),
                ("row pad %", "row_padding", "{:.1%}"), ("(K,N,res) shapes", "n_shapes", "{}"),
                ("MPK queue", "mpk_queue_len", "{}")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--res", default=",".join(map(str, RESOLUTIONS)))
    ap.add_argument("--show", nargs=3, metavar=("MODEL", "BATCH", "RES"), help="print one configuration, write nothing")
    a = ap.parse_args()
    if a.show:
        m, b, r = a.show
        print(table(build(load_arch(m), int(b), int(r))))
        return
    os.makedirs(DAG_DIR, exist_ok=True)
    n = 0
    for m in a.models.split(","):
        arch = load_arch(m)
        for b in map(int, a.batches.split(",")):
            for r in map(int, a.res.split(",")):
                dag = build(arch, b, r)
                p = path(m, b, r)
                text = dag.to_json()
                if not os.path.exists(p) or open(p).read() != text:  # keep the mtime of unchanged files
                    with open(p, "w") as f:
                        f.write(text)
                n += 1
    order = {m: i for i, m in enumerate(MODELS)}
    rows = [json.load(open(os.path.join(DAG_DIR, p)))["summary"] for p in os.listdir(DAG_DIR) if p.endswith(".json")]
    rows.sort(key=lambda s: (order.get(s["model"], len(order)), s["model"], s["batch"], s["res"]))
    md = ["# GEMM DAGs (generated by common/dag.py)", "",
          "One file per configuration: `<model>_b<batch>_r<res>.json`. Conv GFLOP = the real convolutions the "
          "GEMMs stand for; ratio = executed / conv. MPK queue = per-worker task queue length the launch needs "
          f"(default {MPK_QUEUE}).", "",
          "| " + " | ".join(c for c, _, _ in SUMMARY_COLS) + " |", "|" + "---|" * len(SUMMARY_COLS)]
    for s in rows:
        md.append("| " + " | ".join(fmt.format(s[k]) for _, k, fmt in SUMMARY_COLS) + " |")
    with open(os.path.join(DAG_DIR, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[5:]))
    print(f"{n} DAGs built -> {DAG_DIR} (summary.md lists all {len(rows)})")


if __name__ == "__main__":
    main()
