#!/usr/bin/env python3
"""Extract the YOLO26 backbone (layers 0-9) from Ultralytics as a conv DAG: arch/yolo26<model>.json.

  $PY arch/gen_arch.py                 # n, m, l (needs ultralytics from the mirage venv; CPU only)
  $PY arch/gen_arch.py --models n,s

One node per convolution of the real network, named after its Ultralytics module:
  {"name": "model.2.m.0.cv2", "layer": 2, "module": "Bottleneck", "k": 3, "s": 1,
   "in": ["model.2.m.0.cv1"], "cin": 8, "cout": 16, "res": "model.2.cv1.1"}
"in" is the list of tensors concatenated along channels to form the conv input (one entry for a plain
conv), "res" a tensor added to the output (shortcut). A node's output tensor has the node's name.
Left out: BN (folded into the conv weight in deployment), SiLU, the MaxPools of SPPF (cv2 reads cv1's
output n+1 times instead), C2PSA (layer 10) and the head. C3k2.cv1, whose output is chunked in two,
becomes two convs "cv1.0" and "cv1.1" with half of the output channels each.

Check (default on): the real layers 0-9 with BN, SiLU and MaxPool replaced by identity, run in fp64,
must equal the DAG evaluated with F.conv2d on the same weights.
"""
import argparse
import contextlib
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LAST_LAYER = 9


def load_yolo(model):
    """The Ultralytics DetectionModel of yolo26<model>.yaml (random weights).

    Building it sets CUBLAS_WORKSPACE_CONFIG=:4096:8, which makes every cuBLAS call of torch 2.7.1+cu118
    about 70 us slower on the host; the variable is restored afterwards.
    """
    prev = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        from ultralytics import YOLO

        with contextlib.redirect_stdout(io.StringIO()):
            return YOLO(f"yolo26{model}.yaml").model
    finally:
        if prev is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev


def extract(net):
    """-> (nodes, output name, {node name: (nn.Conv2d, first output channel)})."""
    from ultralytics.nn.modules.block import SPPF, Bottleneck, C3k, C3k2
    from ultralytics.nn.modules.conv import Conv

    nodes, chans, weights = [], {"image": 3}, {}

    def conv(name, layer, module, ins, m, res=None, half=None):
        c = m.conv
        k, s = c.kernel_size[0], c.stride[0]
        assert c.groups == 1 and c.bias is None and c.kernel_size == (k, k) and c.stride == (s, s), name
        assert c.padding == (k // 2, k // 2) and c.dilation == (1, 1), name
        assert c.in_channels == sum(chans[i] for i in ins), (name, c.in_channels, ins)
        cout, first = c.out_channels, 0
        if half is not None:  # one half of a conv whose output is chunked (C3k2.cv1)
            cout //= 2
            first = half * cout
            name = f"{name}.{half}"
        node = {"name": name, "layer": layer, "module": module, "k": k, "s": s, "in": list(ins),
                "cin": c.in_channels, "cout": cout}
        if res is not None:
            assert chans[res] == cout, (name, res)
            node["res"] = res
        nodes.append(node)
        chans[name] = cout
        weights[name] = (c, first)
        return name

    def bottleneck(name, layer, x, m):
        t = conv(f"{name}.cv1", layer, "Bottleneck", [x], m.cv1)
        return conv(f"{name}.cv2", layer, "Bottleneck", [t], m.cv2, res=x if m.add else None)

    def c3k(name, layer, x, m):
        a = conv(f"{name}.cv1", layer, "C3k", [x], m.cv1)
        b = conv(f"{name}.cv2", layer, "C3k", [x], m.cv2)
        for i, bn in enumerate(m.m):
            a = bottleneck(f"{name}.m.{i}", layer, a, bn)
        return conv(f"{name}.cv3", layer, "C3k", [a, b], m.cv3)

    x = "image"
    for i, layer in enumerate(net.model[:LAST_LAYER + 1]):
        name = f"model.{i}"
        assert layer.f == -1, (name, layer.f)  # every backbone layer reads the previous one
        if type(layer) is Conv:
            x = conv(name, i, "Conv", [x], layer)
        elif isinstance(layer, C3k2):
            y = [conv(f"{name}.cv1", i, "C3k2", [x], layer.cv1, half=h) for h in (0, 1)]
            for j, blk in enumerate(layer.m):
                if isinstance(blk, C3k):
                    y.append(c3k(f"{name}.m.{j}", i, y[-1], blk))
                elif isinstance(blk, Bottleneck):
                    y.append(bottleneck(f"{name}.m.{j}", i, y[-1], blk))
                else:
                    raise NotImplementedError(f"{name}.m.{j}: {type(blk).__name__}")
            x = conv(f"{name}.cv2", i, "C3k2", y, layer.cv2)
        elif isinstance(layer, SPPF):
            y = conv(f"{name}.cv1", i, "SPPF", [x], layer.cv1)
            x = conv(f"{name}.cv2", i, "SPPF", [y] * (layer.n + 1), layer.cv2, res=x if layer.add else None)
        else:
            raise NotImplementedError(f"{name}: {type(layer).__name__}")
    return nodes, x, weights


def check(net, nodes, output, weights, res=64):
    """Real layers without BN/SiLU/MaxPool vs the DAG with F.conv2d; returns max |diff| / max |ref|."""
    import torch
    import torch.nn.functional as F
    from ultralytics.nn.modules.block import SPPF
    from ultralytics.nn.modules.conv import Conv

    layers = net.model[:LAST_LAYER + 1].double().eval()
    for m in layers.modules():
        if isinstance(m, Conv):
            m.bn, m.act = torch.nn.Identity(), torch.nn.Identity()
        elif isinstance(m, SPPF):
            m.m = torch.nn.Identity()
    img = torch.randn(2, 3, res, res, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = img
        for layer in layers:
            ref = layer(ref)
        vals = {"image": img}
        for n in nodes:
            c, first = weights[n["name"]]
            w = c.weight[first:first + n["cout"]]
            y = F.conv2d(torch.cat([vals[i] for i in n["in"]], 1), w, stride=n["s"], padding=n["k"] // 2)
            vals[n["name"]] = y + vals[n["res"]] if "res" in n else y
    out = vals[output]
    assert out.shape == ref.shape, (out.shape, ref.shape)
    return ((out - ref).abs().max() / ref.abs().max()).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="n,m,l")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    import torch
    import ultralytics

    for model in a.models.split(","):
        net = load_yolo(model)
        nodes, output, weights = extract(net)
        n_conv2d = sum(isinstance(m, torch.nn.Conv2d) for m in net.model[:LAST_LAYER + 1].modules())
        n_c3k2 = sum(type(m).__name__ == "C3k2" for m in net.model[:LAST_LAYER + 1])
        assert len(nodes) == n_conv2d + n_c3k2, (len(nodes), n_conv2d, n_c3k2)  # every Conv2d, cv1 split in two
        rel = None if a.no_check else check(net, nodes, output, weights)
        assert rel is None or rel < 1e-10, rel
        doc = {
            "model": model,
            "source": f"ultralytics {ultralytics.__version__}, yolo26{model}.yaml, layers 0-{LAST_LAYER}",
            "backbone_yaml": net.yaml["backbone"][:LAST_LAYER + 1],
            "scale": net.yaml["scales"][model],
            "omitted": ["BatchNorm (folded into the conv weight in deployment)", "SiLU", "SPPF MaxPool x3 (identity)",
                        "layer 10 (C2PSA) and the head"],
            "check": {"real_layers_vs_dag_rel_diff": rel},
            "input": {"name": "image", "C": 3},
            "output": output,
            "convs": nodes,
        }
        path = os.path.join(HERE, f"yolo26{model}.json")
        with open(path, "w") as f:
            f.write(to_json(doc))
        print(f"yolo26{model}: {len(nodes)} convs ({n_conv2d} Conv2d), output {output} "
              f"[{nodes[-1]['cout']} ch], check rel diff {rel if rel is None else f'{rel:.1e}'} -> {path}")


def to_json(doc):
    """JSON with one list item (conv, yaml row) per line."""
    parts = []
    for k, v in doc.items():
        if isinstance(v, list) and k != "scale":
            items = ",\n".join("  " + json.dumps(x) for x in v)
            parts.append(f' {json.dumps(k)}: [\n{items}\n ]')
        else:
            parts.append(f" {json.dumps(k)}: {json.dumps(v)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


if __name__ == "__main__":
    main()
