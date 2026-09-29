"""MPK environment and build helpers (taken from test/0928/tools/mpk_common.py, only what 02_mpk uses).

Call setup() before importing torch or mirage so that PATH (nvcc), the GPU selection and the
annotated-graph dump flag are in place.
"""
import contextlib
import os
import sys
import time

MIRAGE_HOME = os.environ.get("MIRAGE_HOME", "/workspace/toyota/mirage")
# nvcc for the generated megakernel must be CUDA >= 12; /usr/local/cuda is 11.8 on this machine.
MPK_CUDA_HOME = os.environ.get("MPK_CUDA_HOME") or next(
    (d for d in ("/opt/cuda-12.0-mpk", "/usr/local/cuda-12.0") if os.path.exists(os.path.join(d, "bin", "nvcc"))),
    "/usr/local/cuda-12.0")


def setup():
    """Environment for building and running MPK megakernels (see env.sh for the reasons)."""
    os.environ["MIRAGE_HOME"] = MIRAGE_HOME
    if os.path.exists(os.path.join(MPK_CUDA_HOME, "bin", "nvcc")):
        os.environ["MPK_CUDA_HOME"] = MPK_CUDA_HOME
        cuda_bin = os.path.join(MPK_CUDA_HOME, "bin")
        if not os.environ.get("PATH", "").startswith(cuda_bin):
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        # The generated launcher links libcudart.so.12; load it now so the loader finds it
        # even when LD_LIBRARY_PATH was not set before Python started.
        import ctypes
        ctypes.CDLL(os.path.join(MPK_CUDA_HOME, "lib64", "libcudart.so.12"), mode=ctypes.RTLD_GLOBAL)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("MPK_GPU", "6")
    os.environ.setdefault("MIRAGE_DUMP_ANNOTATED_GRAPH", "1")
    p = os.path.join(MIRAGE_HOME, "python")
    if p not in sys.path:
        sys.path.insert(0, p)
    # mirage picks -std=c++20 whenever g++ accepts it. With gcc 11 and nvcc 12.0 that breaks deps/json
    # (iteration_proxy.hpp: "expected initializer before '<'"), so pin c++17. Importing mirage here
    # does not initialize CUDA.
    import mirage.mpk.persistent_kernel as mpk_pk
    mpk_pk._detect_cxx_standard = lambda: "-std=c++17"


@contextlib.contextmanager
def capture_fd(fd, path):
    """Redirect an OS-level file descriptor (e.g. 2 = stderr) into a file.

    C++ code (the AnnotatedGraph dump) and nvcc subprocesses write straight to the descriptor, so
    sys.stderr redirection is not enough.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(fd)
    with open(path, "ab") as f:
        os.dup2(f.fileno(), fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved, fd)
        os.close(saved)


def extract_annotated(log_path, out_path):
    """Copy the first 'AnnotatedGraph:' block of a log into its own file."""
    if not os.path.exists(log_path):
        return None
    block, inside = [], False
    for line in open(log_path, errors="replace").read().splitlines():
        if line.startswith("AnnotatedGraph: "):
            if block:
                break
            block, inside = [line], True
            continue
        if inside:
            if not line.startswith("  "):
                inside = False
                continue
            block.append(line)
            if line.startswith("  ordered_layers:"):
                inside = False
    if not block:
        return None
    with open(out_path, "w") as f:
        f.write("\n".join(block) + "\n")
    return out_path


def compile_and_export(pk, out_dir):
    """pk.compile(output_dir=out_dir) with stderr captured to out_dir/compile.log.

    Leaves task_graph_rank0.json, test_rank0.cu, the launcher .so, kernel_metadata_rank0.json and
    annotated.txt in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "compile.log")
    if os.path.exists(log):
        os.remove(log)
    t0 = time.time()
    try:
        with capture_fd(2, log):
            pk.compile(output_dir=out_dir)
    except Exception:
        if os.path.exists(log):
            sys.stderr.write(open(log, errors="replace").read()[-6000:])
        raise
    dt = time.time() - t0
    extract_annotated(log, os.path.join(out_dir, "annotated.txt"))
    print(f"[compile] {dt:.1f} s -> {out_dir}")
    return dt


def mpk_params(batch, requests=None, **kw):
    """PersistentKernel test_mode parameters.

    page_size/max_num_pages/max_seq_length are raised from the defaults (1/1/1): the offline scheduler
    allocates ceil(tokens / page_size) pages per request and reads the prompt from tokens[0, :batch],
    which would overrun the 1-entry defaults.
    """
    import mirage
    from mirage.mpk.persistent_kernel import PersistentKernel

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=num_workers, num_local_schedulers=num_schedulers,
             mpi_rank=0, world_size=1, max_num_batched_tokens=batch,
             max_num_batched_requests=requests or batch, page_size=64, max_num_pages=4,
             max_seq_length=64)
    p.update(kw)
    return p
