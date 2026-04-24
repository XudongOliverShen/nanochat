"""Reader for per-token activation / gradient norm dumps.

These files are produced by ``GradientBiasMonitor`` in
``scripts_dev/exp1_base_train.py`` during training. Each training step emits
(across all DDP ranks and grad-accumulation micro-steps) one batch of
per-token hidden-state norms and per-(token, head) Q/K/V norms; the monitor
batches ``--monitor-steps-per-file`` consecutive steps into one pair of npz
files:

    <logs_dir>/norms/step_{first}-{last}_hidden.npz
    <logs_dir>/norms/step_{first}-{last}_attn.npz

This module loads those files, de-quantizes them back to float32, and
concatenates across files to cover a requested global-step range.

Data layout (what each returned array means)
============================================
Each "sample" in the stacked arrays is identified by the implicit axis tuple
``(s, a, r, b, l, ...)`` with sizes:

    S = # global_steps in the returned range   A = grad_accum_steps
    R = world_size (# DDP ranks)               B = device_batch_size
    L = n_layer                                T = seq_len
    H_q = n_head (query heads)                 H_kv = n_kv_head (GQA kv heads)

Axis meaning:
    s -> global_step = result.global_steps[s]
    a -> grad-accumulation micro-step index (0 .. A-1)
    r -> DDP rank (0 .. R-1)
    b -> per-rank sample index within the device batch (0 .. B-1)
    l -> transformer layer index (0 .. L-1); result.layer_types[l] tells
         you whether layer l is "full_attention" or "sliding_attention"

Hidden arrays carry the per-token activation/gradient norm of the block
output (post-residual hidden state, magnitude over the embedding dim):

    HiddenNorms.act [s,a,r,b,l,t]  : ||hidden[s,a,r,b,l,t,:]||_2   (fp32)
    HiddenNorms.grad[s,a,r,b,l,t]  : ||dL/dhidden ...||_2          (fp32)

Attention arrays carry the per-head norm of each token's Q / K / V
projection (magnitude over head_dim):

    AttnNorms.q_act [s,a,r,b,l,t,h]  : ||q_proj_out[...,h,:]||_2    (fp32)
    AttnNorms.k_act [s,a,r,b,l,t,h]  : ||k_proj_out[...,h,:]||_2    (fp32)
    AttnNorms.v_act [s,a,r,b,l,t,h]  : ||v_proj_out[...,h,:]||_2    (fp32)
    AttnNorms.q_grad / k_grad / v_grad : upstream gradient norms, same shapes

Lossy note
----------
The on-disk format is uint8 quantized per-sample with the top
``outlier_pct`` fraction stored losslessly in fp32. After de-quantization
values are fp32 but only ~8-bit precise except at the preserved outlier
positions. See ``outlier_pct`` in the returned metadata.

Usage examples
==============

1. Read hidden norms only, steps 0..3 inclusive::

     from plot_norm.read_norms import read_hidden
     h = read_hidden("logs/speedrun_04241621/norms", 0, 3)
     print(h.act.shape)           # (S, A, R, B, L, T) fp32
     print(h.global_steps)        # e.g. array([0,1,2,3])
     print(h.layer_types)         # ['full_attention', 'sliding_attention', ...]
     # mean hidden-norm trajectory for layer 5, averaged over (a,r,b,t):
     import numpy as np
     traj = h.act[:, :, :, :, 5, :].mean(axis=(1,2,3,4))   # (S,)

2. Read attention norms only::

     from plot_norm.read_norms import read_attn
     a = read_attn("logs/speedrun_04241621/norms", 0, 3)
     print(a.q_act.shape)   # (S, A, R, B, L, T, H_q)
     print(a.k_act.shape)   # (S, A, R, B, L, T, H_kv)
     # per-head mean Q-norm for layer 0 at step 0:
     qmean = a.q_act[0].mean(axis=(0,1,2,3))    # (H_q,)

3. Read both modalities over the same range::

     from plot_norm.read_norms import read_both
     h, a = read_both("logs/speedrun_04241621/norms", 0, 3)

4. Single step is a range of length 1::

     h = read_hidden(".../norms", step_start=2, step_end=2)

Running this file directly prints a summary for a default path::

     python -m plot_norm.read_norms --norms-dir logs/.../norms --start 0 --end 3
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Return containers
# ---------------------------------------------------------------------------

@dataclass
class HiddenNorms:
    """Dequantized hidden-state norms over a global-step range.

    Shapes use symbols (S, A, R, B, L, T) described in the module docstring.
    """
    act: np.ndarray            # (S, A, R, B, L, T) float32   activation norms
    grad: np.ndarray           # (S, A, R, B, L, T) float32   gradient norms
    global_steps: np.ndarray   # (S,) int32                   step id per axis-0 slice
    layer_types: List[str]     # length L
    n_elements: int            # embedding dim C
    seq_len: int               # T
    device_batch_size: int     # B
    world_size: int            # R
    grad_accum_steps: int      # A
    outlier_pct: float         # fraction of values kept losslessly per sample


@dataclass
class AttnNorms:
    """Dequantized Q / K / V per-head norms over a global-step range.

    Shapes use symbols (S, A, R, B, L, T, H_*) described in the module
    docstring. Q has head axis size H_q; K and V share head axis size H_kv.
    """
    q_act: np.ndarray          # (S, A, R, B, L, T, H_q)
    q_grad: np.ndarray
    k_act: np.ndarray          # (S, A, R, B, L, T, H_kv)
    k_grad: np.ndarray
    v_act: np.ndarray          # (S, A, R, B, L, T, H_kv)
    v_grad: np.ndarray
    global_steps: np.ndarray   # (S,) int32
    layer_types: List[str]     # length L
    n_head: int                # H_q
    n_kv_head: int             # H_kv
    head_dim: int
    seq_len: int
    device_batch_size: int
    world_size: int
    grad_accum_steps: int
    outlier_pct: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(r"step_(\d+)-(\d+)_(hidden|attn)\.npz$")


def _discover_files(norms_dir: str, modality: str,
                    step_start: int, step_end: int) -> List[Tuple[int, int, str]]:
    """Return sorted [(first, last, path), ...] for files whose [first,last]
    range overlaps [step_start, step_end]. Raises if nothing matches."""
    assert modality in ("hidden", "attn")
    if step_start > step_end:
        raise ValueError(f"step_start ({step_start}) > step_end ({step_end})")

    out = []
    for path in sorted(glob.glob(os.path.join(norms_dir, f"step_*_{modality}.npz"))):
        m = _FNAME_RE.search(os.path.basename(path))
        if m is None or m.group(3) != modality:
            continue
        f, l = int(m.group(1)), int(m.group(2))
        if l < step_start or f > step_end:
            continue
        out.append((f, l, path))
    if not out:
        raise FileNotFoundError(
            f"no {modality} files overlap [{step_start},{step_end}] in {norms_dir}")
    out.sort(key=lambda t: t[0])
    return out


def _dequantize(q: np.ndarray, scale: np.ndarray, mn: np.ndarray,
                outlier_idx: np.ndarray, outlier_val: np.ndarray) -> np.ndarray:
    """Invert the per-sample uint8 quantization.

    Shapes:
      q           : sample_shape + data_shape        uint8
      scale, mn   : sample_shape                     float32
      outlier_idx : sample_shape + (K,)              int32, flat indices into data_shape
      outlier_val : sample_shape + (K,)              float32

    Returns fp32 array with shape equal to q.shape. Formula per sample:
      val = mn + q.astype(f32) * scale
      val.flat[outlier_idx] = outlier_val
    """
    n_data = q.ndim - scale.ndim
    assert n_data >= 1
    bcast = scale.shape + (1,) * n_data
    val = mn.reshape(bcast) + q.astype(np.float32) * scale.reshape(bcast)

    # splice outliers into the flattened data-axes view
    sample_shape = scale.shape
    data_flat = int(np.prod(q.shape[-n_data:]))
    flat_view = val.reshape(sample_shape + (data_flat,))
    np.put_along_axis(flat_view, outlier_idx, outlier_val.astype(np.float32), axis=-1)
    return flat_view.reshape(q.shape)


def _decode_layer_types(d) -> List[str]:
    legend = [str(s) for s in d["layer_type_legend"]]
    return [legend[i] for i in d["layer_types"].tolist()]


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------

def read_hidden(norms_dir: str, step_start: int, step_end: int) -> HiddenNorms:
    """Read hidden-state activation / gradient norms for steps in [start, end].

    Parameters
    ----------
    norms_dir  : path to the ``<logs_dir>/norms`` directory.
    step_start : inclusive lower bound on global_step.
    step_end   : inclusive upper bound on global_step.

    Returns
    -------
    HiddenNorms  with ``.act`` and ``.grad`` of shape (S, A, R, B, L, T),
    where S = number of steps actually present in [start, end].
    """
    files = _discover_files(norms_dir, "hidden", step_start, step_end)

    act_parts, grad_parts, steps_parts = [], [], []
    meta = None
    for _f, _l, path in files:
        d = np.load(path)
        gs = d["global_steps"]
        keep = (gs >= step_start) & (gs <= step_end)
        if not keep.any():
            continue

        act = _dequantize(d["act_q"], d["act_scale"], d["act_min"],
                          d["act_outlier_idx"], d["act_outlier_val"])
        grd = _dequantize(d["grad_q"], d["grad_scale"], d["grad_min"],
                          d["grad_outlier_idx"], d["grad_outlier_val"])
        act_parts.append(act[keep])
        grad_parts.append(grd[keep])
        steps_parts.append(gs[keep])

        if meta is None:
            meta = dict(
                layer_types=_decode_layer_types(d),
                n_elements=int(d["n_elements"]),
                seq_len=int(d["seq_len"]),
                device_batch_size=int(d["device_batch_size"]),
                world_size=int(d["world_size"]),
                grad_accum_steps=int(d["grad_accum_steps"]),
                outlier_pct=float(d["outlier_pct"]),
            )

    if not act_parts:
        raise FileNotFoundError(
            f"no hidden records for global_step in [{step_start},{step_end}]")

    return HiddenNorms(
        act=np.concatenate(act_parts, axis=0),
        grad=np.concatenate(grad_parts, axis=0),
        global_steps=np.concatenate(steps_parts, axis=0).astype(np.int32),
        **meta,
    )


def read_attn(norms_dir: str, step_start: int, step_end: int) -> AttnNorms:
    """Read Q/K/V per-head activation / gradient norms for steps in [start, end].

    Returns
    -------
    AttnNorms with ``q_act`` / ``q_grad`` of shape (S, A, R, B, L, T, H_q)
    and ``k_act`` / ``k_grad`` / ``v_act`` / ``v_grad`` of shape
    (S, A, R, B, L, T, H_kv). The on-disk k/v pair is split apart for you.
    """
    files = _discover_files(norms_dir, "attn", step_start, step_end)

    q_act_p, q_grad_p = [], []
    k_act_p, k_grad_p = [], []
    v_act_p, v_grad_p = [], []
    steps_parts = []
    meta = None
    for _f, _l, path in files:
        d = np.load(path)
        gs = d["global_steps"]
        keep = (gs >= step_start) & (gs <= step_end)
        if not keep.any():
            continue

        q_act  = _dequantize(d["q_act_q"],  d["q_act_scale"],  d["q_act_min"],
                             d["q_act_outlier_idx"],  d["q_act_outlier_val"])
        q_grad = _dequantize(d["q_grad_q"], d["q_grad_scale"], d["q_grad_min"],
                             d["q_grad_outlier_idx"], d["q_grad_outlier_val"])
        kv_act  = _dequantize(d["kv_act_q"],  d["kv_act_scale"],  d["kv_act_min"],
                              d["kv_act_outlier_idx"],  d["kv_act_outlier_val"])
        kv_grad = _dequantize(d["kv_grad_q"], d["kv_grad_scale"], d["kv_grad_min"],
                              d["kv_grad_outlier_idx"], d["kv_grad_outlier_val"])
        # kv axis 5: 0=k, 1=v (see GradientBiasMonitor.kv_legend)
        assert list(d["kv_legend"]) == ["k", "v"], f"unexpected kv_legend: {d['kv_legend']}"
        k_act,  v_act  = kv_act[:, :, :, :, :, 0], kv_act[:, :, :, :, :, 1]
        k_grad, v_grad = kv_grad[:, :, :, :, :, 0], kv_grad[:, :, :, :, :, 1]

        q_act_p.append(q_act[keep]);   q_grad_p.append(q_grad[keep])
        k_act_p.append(k_act[keep]);   k_grad_p.append(k_grad[keep])
        v_act_p.append(v_act[keep]);   v_grad_p.append(v_grad[keep])
        steps_parts.append(gs[keep])

        if meta is None:
            meta = dict(
                layer_types=_decode_layer_types(d),
                n_head=int(d["n_head"]),
                n_kv_head=int(d["n_kv_head"]),
                head_dim=int(d["head_dim"]),
                seq_len=int(d["seq_len"]),
                device_batch_size=int(d["device_batch_size"]),
                world_size=int(d["world_size"]),
                grad_accum_steps=int(d["grad_accum_steps"]),
                outlier_pct=float(d["outlier_pct"]),
            )

    if not q_act_p:
        raise FileNotFoundError(
            f"no attn records for global_step in [{step_start},{step_end}]")

    cat = lambda parts: np.concatenate(parts, axis=0)
    return AttnNorms(
        q_act=cat(q_act_p),   q_grad=cat(q_grad_p),
        k_act=cat(k_act_p),   k_grad=cat(k_grad_p),
        v_act=cat(v_act_p),   v_grad=cat(v_grad_p),
        global_steps=cat(steps_parts).astype(np.int32),
        **meta,
    )


def read_both(norms_dir: str, step_start: int,
              step_end: int) -> Tuple[HiddenNorms, AttnNorms]:
    """Convenience: read both modalities for the same step range."""
    return (read_hidden(norms_dir, step_start, step_end),
            read_attn(norms_dir, step_start, step_end))


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _print_summary(norms_dir: str, start: int, end: int) -> None:
    h, a = read_both(norms_dir, start, end)
    print(f"[hidden] steps={h.global_steps.tolist()}  act={h.act.shape}  grad={h.grad.shape}")
    print(f"         layer_types={h.layer_types}")
    print(f"         T={h.seq_len} B={h.device_batch_size} R={h.world_size} "
          f"A={h.grad_accum_steps} C={h.n_elements} outlier_pct={h.outlier_pct}")
    print(f"         act[:,:,:,:,:, :].mean={h.act.mean():.4f}  "
          f"grad.mean={h.grad.mean():.4e}")
    print(f"[attn]   q_act={a.q_act.shape}  k_act={a.k_act.shape}  v_act={a.v_act.shape}")
    print(f"         H_q={a.n_head} H_kv={a.n_kv_head} head_dim={a.head_dim}")
    print(f"         q_act.mean={a.q_act.mean():.4f}  k_act.mean={a.k_act.mean():.4f}  "
          f"v_act.mean={a.v_act.mean():.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--norms-dir", required=True,
                    help="path to the <logs_dir>/norms directory")
    ap.add_argument("--start", type=int, required=True, help="inclusive global_step lower bound")
    ap.add_argument("--end",   type=int, required=True, help="inclusive global_step upper bound")
    args = ap.parse_args()
    _print_summary(args.norms_dir, args.start, args.end)
