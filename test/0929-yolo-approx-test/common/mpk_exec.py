"""A GEMM DAG (common/dag.py) as one MPK PersistentKernel on sm_86.

Every GEMM is one MPK `linear` (or `linear_with_residual`) op with grid (N/64, M/16): one task per
16 x 64 output tile, cutlass kernel (use_cutlass_kernel=True, which needs K % 128 == 0). Buffers are
torch tensors attached in their producer's [M, N] layout; a GEMM that reads another shape gets a view.

MPK behaviour this relies on (mirage mpk 1f3338f9, unmodified; probed on this machine):
  * A same-rank reshape view (2-D -> 2-D) keeps the parent's strides (src/kernel/view.cc
    set_view_strides): [1024, 32] -> [256, 128] is read with row stride 32, i.e. wrong. Reshapes go
    through a 3-D view ([M, K/C, C] -> [M, K]); a rank change gets row-major strides. narrow is fine.
  * View edges are coarse barriers in the dependency analysis (annotated_graph.cc): a GEMM that reads
    through a view waits for its whole producer; same-layout edges get per-tile events.
  * A launch queues all its tasks at start, 64 workers x 8192 entries (persistent_kernel.cuh). Larger
    DAGs are compiled against a patched copy of that header (enable_worker_queue).
  * The runtime's gpu_malloc does not check cudaMalloc: a failed allocation shows up later as an illegal
    memory access. check_cuda_error() right after initialization reports it instead.
"""
import ctypes
import os
import re

import dag as D

MIRAGE_HOME = os.environ.get("MIRAGE_HOME", "/workspace/toyota/mirage")


def enable_worker_queue(length, overlay_dir):
    """Compile with a longer per-worker task queue without touching the mirage tree.

    A patched copy of persistent_kernel.cuh goes into overlay_dir, and its -I is put in front of mirage's
    include paths (the header's own "..." includes still resolve to the mirage tree).
    """
    import mirage.mpk.persistent_kernel as mpk_pk

    src = os.path.join(MIRAGE_HOME, "include", "mirage", "persistent_kernel", "persistent_kernel.cuh")
    text = open(src).read()
    pat = f"global_runtime_config.per_worker_queue_len = {D.MPK_QUEUE};"
    assert text.count(pat) == 1, "persistent_kernel.cuh changed; update enable_worker_queue()"
    os.makedirs(overlay_dir, exist_ok=True)
    with open(os.path.join(overlay_dir, "persistent_kernel.cuh"), "w") as f:
        f.write(text.replace(pat, f"global_runtime_config.per_worker_queue_len = {length};"))
    orig = getattr(mpk_pk.get_compile_command, "__wrapped__", mpk_pk.get_compile_command)

    def wrapped(*a, **kw):
        cmd = orig(*a, **kw)
        first_inc = next(i for i, c in enumerate(cmd) if c.startswith("-I"))
        return cmd[:first_inc] + [f"-I{overlay_dir}"] + cmd[first_inc:]

    wrapped.__wrapped__ = orig
    mpk_pk.get_compile_command = wrapped


def check_cuda_error(where):
    """Raise if a CUDA runtime call of the MPK launcher failed since the last check (same libcudart.so.12)."""
    home = os.environ.get("MPK_CUDA_HOME", "/usr/local/cuda-12.0")
    lib = ctypes.CDLL(os.path.join(home, "lib64", "libcudart.so.12"))
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    err = lib.cudaGetLastError()
    if err != 0:
        raise RuntimeError(f"{where}: CUDA error {err} ({lib.cudaGetErrorString(err).decode()})")


def safe(name):
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


class MPKDag:
    """Attaches the DAG's tensors to a PersistentKernel and registers one linear op per GEMM."""

    def __init__(self, pk, dag, weights, x, outputs):
        """outputs: GEMM name -> torch [rows, C] buffer that MPK writes (dag.alloc_outputs)."""
        from mirage.core import CyTBGraph
        from mirage.kernel import TBGraph

        self.pk, self.dag = pk, dag
        self.dt = {dag.input: pk.attach_input(x, name="in_" + safe(dag.input))}  # buffer -> DTensor, its layout
        for gm in dag.gemms:
            self.dt[gm.name] = pk.attach_input(outputs[gm.name].view(gm.M, gm.N), name="act_" + safe(gm.name))
        self.n_views = 0
        for gm in dag.gemms:
            A = self.view(gm.a, gm.M, gm.K)
            W = pk.attach_input(weights[gm.name], name="w_" + safe(gm.name))
            O = self.dt[gm.name]
            tb = TBGraph(CyTBGraph((gm.N // D.NQ, gm.M // D.MQ, 1), (128, 1, 1), 1, 64))
            tb.new_input(A, (-1, 0, -1), 1, True)
            tb.new_input(W, (0, -1, -1), 1, True)
            ins = [A, W]
            if gm.res is not None:
                R = self.view(gm.res, gm.M, gm.N)
                tb.new_input(R, (1, 0, -1), -1, True)
                ins.append(R)
            tb.new_input(O, (1, 0, -1), -1, True)
            pk.kn_graph.customized(ins + [O], tb)
            pk.kn_graph.register_task(tb, "linear_with_residual" if gm.res is not None else "linear")

    def view(self, buf, M, K):
        """DTensor [M, K] over the first M*K elements of a buffer."""
        t = self.dt[buf]
        Mp, Np = self.dag.layout(buf)
        if (M, K) == (Mp, Np):
            return t  # same layout: per-tile dependencies
        self.n_views += 1
        n = M * K
        assert n % Np == 0 and n <= Mp * Np, (buf, M, K, Mp, Np)
        if n < Mp * Np:
            t = self.pk.narrow(t, 0, 0, n // Np)
        C = self.dag.buffers[buf].C
        t3 = self.pk.view(t, [M, K // C, C])  # rank change -> row-major strides (see the module doc)
        return self.pk.view(t3, [M, K])
