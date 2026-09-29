"""Run a GEMM DAG (common/dag.py) with torch.mm / torch.addmm (cuBLAS) into preallocated buffers."""
import torch

import dag as D


class TorchDag:
    """One cuBLAS call per GEMM; operand views are made once, so forward() is capturable as a CUDA graph."""

    def __init__(self, dag, weights, x, outputs=None):
        self.dag = dag
        self.bufs = {dag.input: x, **(outputs if outputs is not None else D.alloc_outputs(dag, x))}
        self.steps = []
        for gm in dag.gemms:
            A = self.bufs[gm.a].view(-1)[: gm.M * gm.K].view(gm.M, gm.K)  # first M*K elements of A's buffer
            out = self.bufs[gm.name].view(gm.M, gm.N)
            R = None if gm.res is None else self.bufs[gm.res].view(gm.M, gm.N)
            self.steps.append((A, weights[gm.name].t(), R, out))

    def forward(self):
        for A, Wt, R, out in self.steps:
            if R is None:
                torch.mm(A, Wt, out=out)
            else:
                torch.addmm(R, A, Wt, out=out)
        return self.bufs[self.dag.output]


def reference(dag, weights, x):
    """The same DAG in fp32 (buffers of its own), already run."""
    ref = TorchDag(dag, {k: v.float() for k, v in weights.items()}, x.float())
    ref.forward()
    torch.cuda.synchronize()
    return ref
