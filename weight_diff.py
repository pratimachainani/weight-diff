"""weight_diff.py: compare a model checkpoint against its declared base, neuron by neuron.

Microsoft's guidance (10 Nov 2026): legitimate fine-tunes and quantizations change weights
"broadly and diffusely"; implanted edits concentrate on a few weight vectors that change a lot.
This script shows which one a checkpoint looks like.

Usage (Colab or any machine with enough RAM for two copies of the model):
    pip install -q torch safetensors huggingface_hub
    python weight_diff.py <base_model_id> <other_model_id>
    python weight_diff.py <base_model_id> <other_model_id> --spike 7   # demo: plant 7 synthetic outlier neurons

--spike adds a large random change to N rows of the LAST layer's MLP gate projection in memory only.
It is a visual stand-in for a concentrated edit. It is NOT a backdoor and changes nothing on disk.

Scope: both checkpoints must be safetensors in the same floating-point format (bf16/fp16/fp32).
Integer or FP8 quantized copies (GPTQ, AWQ, FP8) are not compared correctly: dequantize first.
MLP down projections are scored by column (one column = one neuron's output); other matrices by row.
"""
import argparse, glob, os, re
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def load(model_id):
    path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        sd.update(load_file(f))
    return sd


def last_layer_gate(names):
    best, idx = None, -1
    for n in names:
        m = re.search(r"layers\.(\d+)\.mlp\.gate_proj\.weight$", n)
        if m and int(m.group(1)) > idx:
            best, idx = n, int(m.group(1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("other")
    ap.add_argument("--spike", type=int, default=0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--include-embeddings", action="store_true",
                    help="also compare embedding and output-head rows (new chat tokens make these noisy)")
    a = ap.parse_args()

    print(f"loading base  {a.base}"); base = load(a.base)
    print(f"loading other {a.other}"); other = load(a.other)
    common = [n for n in base if n in other and base[n].ndim == 2 and base[n].shape == other[n].shape
              and (a.include_embeddings or not re.search(r"embed|lm_head", n))]
    print(f"comparing {len(common)} weight matrices")

    if a.spike:
        g = last_layer_gate(common)
        if g is None:
            raise SystemExit("--spike: no 'layers.N.mlp.gate_proj.weight' matrix found in this model (fused or MoE layout). Run without --spike.")
        torch.manual_seed(7)
        rows = torch.randperm(other[g].shape[0])[: a.spike]
        w = other[g].float().clone()
        w[rows] += 8 * w[rows].norm(dim=1, keepdim=True) * torch.randn_like(w[rows]) / w.shape[1] ** 0.5
        other[g] = w.to(other[g].dtype)
        print(f"[demo] planted {a.spike} synthetic outlier rows in {g}: {sorted(rows.tolist())}")

    if not common:
        raise SystemExit("No matching safetensors weight matrices. Both models need safetensors files with the same layer names and shapes.")
    rel = []  # (relative change of one neuron's weights, tensor name, label)
    for n in common:
        b, o = base[n].float(), other[n].float()
        dim, kind = (0, "col") if re.search(r"down_proj|w2\.weight", n) else (1, "row")
        bn = b.norm(dim=dim)
        floor = bn.median() * 0.05 + 1e-8   # neurons with near-zero base weights don't explode the score
        d = (o - b).norm(dim=dim) / torch.maximum(bn, floor)
        for i, v in enumerate(d.tolist()):
            rel.append((v, n, f"{kind} {i}"))
    vals = torch.tensor([r[0] for r in rel])
    med = vals.median().item()
    mad = (vals - med).abs().median().item() + 1e-12
    p99 = vals.quantile(0.99).item() if len(vals) < 16_000_000 else sorted(vals.tolist())[int(len(vals) * 0.99)]
    print(f"\nneurons compared: {len(vals):,}")
    print(f"median relative change per neuron: {med:.4f} · 99th percentile: {p99:.4f} · max: {vals.max().item():.4f} · typical spread (MAD): {mad:.6f}")
    if mad < 1e-9:
        print("note: almost nothing changed between the two checkpoints, so the z scores below are not meaningful.")
    changed = (vals > 1e-6).float().mean().item() * 100
    print(f"neurons that changed at all: {changed:.1f}%")

    rel.sort(reverse=True)
    print(f"\ntop {a.top} neurons (robust z = how many typical spreads above the median):")
    flagged = 0
    for v, n, i in rel[: a.top]:
        z = (v - med) / (1.4826 * mad)
        flag = "  <-- outlier" if z > 50 else ""
        flagged += z > 50
        print(f"  {v:8.4f}  z={z:10.1f}  {n} {i}{flag}")
    print("\nverdict:", "CONCENTRATED: a few neurons changed far more than the rest. Audit before use."
          if flagged else "DIFFUSE: changes are spread out, the pattern of a normal fine-tune or quantization.")
    print("A diffuse result is not proof of safety. A concentrated one is a reason to stop and look.")


if __name__ == "__main__":
    main()
