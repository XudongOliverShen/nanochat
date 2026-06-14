"""Reader for the *multi-stream* per-token activation / gradient norm dumps.

This is the successor to ``read_norms.py``. It reads the format written by the
updated ``GradientBiasMonitor`` in ``scripts_dev/exp2_base_train.py``, where the
"hidden" side now records **four** per-token residual-stream points per layer
instead of one. A ``stream`` axis (size ``HS=4``) sits between the batch axis and
the layer axis, and the file carries a ``stream_legend`` naming each slot:

    ["block_in", "attn_in", "attn_out", "block_out"]

For a transformer block ``x = x + attn(norm(x)); x = x + mlp(norm(x))``:

    block_in  = x            (block input)
    attn_in   = norm(x)      (attention input)
    attn_out  = attn(norm(x))(attention output)
    block_out = block output (post attn + mlp residual)

The **attention (Q/K/V)** side is byte-identical to the old format, so
:func:`read_attn` here is the same as in ``read_norms.py``.

The monitor records a fwd/bwd pass every ``--monitor-record-every-k-steps`` global
steps (default 20), so recorded ``global_step`` values are typically *sparse*. Each
recorded step contributes (across all DDP ranks and grad-accumulation micro-steps)
one batch of norms; the monitor batches ``--monitor-steps-per-file`` consecutive
*recorded* steps into one pair of npz files:

    <logs_dir>/norms/step_{first}-{last}_hidden.npz
    <logs_dir>/norms/step_{first}-{last}_attn.npz

``{first}``/``{last}`` are the first/last *recorded* global_step ids in the file
(in-between ids may be skipped, stride = ``record_every_k_steps``). The authoritative
list of recorded ids lives inside the file under ``global_steps``.

This module loads those files, de-quantizes them back to float32, and concatenates
across files to cover a requested global-step range. ``skip_every`` subsamples the
recorded steps in range (every k-th one), multiplying the on-disk stride.

Data layout (what each returned array means)
============================================
Each "sample" in the stacked hidden arrays is identified by the axis tuple
``(s, a, r, b, hs, l, t)`` with sizes:

    S  = # recorded global_steps in range (may be sparse)
    A  = grad_accum_steps
    R  = world_size (# DDP ranks)              B  = device_batch_size
    HS = # hidden streams (= len(stream_legend), 4)
    L  = n_layer                               T  = seq_len

Axis meaning:
    s  -> global_step = result.global_steps[s]   (sparse-safe; always look up)
    a  -> grad-accumulation micro-step index
    r  -> DDP rank
    b  -> per-rank sample index within the device batch
    hs -> hidden stream; result.stream_legend[hs] names it
    l  -> transformer layer index; result.layer_types[l] is full/sliding attention

Hidden arrays carry the per-token activation/gradient norm (magnitude over the
embedding dim) of the chosen residual-stream point:

    HiddenNorms.act [s,a,r,b,hs,l,t]  : ||stream_act[...]||_2   (fp32)
    HiddenNorms.grad[s,a,r,b,hs,l,t]  : ||dL/dstream ...||_2    (fp32)

Use :meth:`HiddenNorms.stream` to pull a single stream as a 6-D
``(S, A, R, B, L, T)`` array — the exact shape the legacy plotting code expects.

Attention arrays (unchanged from the old format) carry the per-head norm of each
token's Q / K / V projection (magnitude over head_dim):

    AttnNorms.q_act [s,a,r,b,l,t,h]  : ||q_proj_out[...,h,:]||_2    (fp32)
    AttnNorms.k_act / v_act          : K / V, head axis size H_kv
    AttnNorms.q_grad / k_grad / v_grad : upstream gradient norms, same shapes

Lossy note
----------
The on-disk format is uint8 quantized per-sample with the top ``outlier_pct``
fraction stored losslessly in fp32. After de-quantization values are fp32 but only
~8-bit precise except at the preserved outlier positions.

Usage examples
==============

1. Read hidden norms (all four streams) for steps in [0, 40]::

     from plot_norm.read_norms_new import read_hidden
     h = read_hidden("logs/run/norms", 0, 40)
     print(h.act.shape)        # (S, A, R, B, HS, L, T) fp32
     print(h.stream_legend)    # ['block_in','attn_in','attn_out','block_out']
     attn_out = h.stream("attn_out")    # (S, A, R, B, L, T) — one stream
     block_in = h.stream("block_in")

2. Read attention norms only::

     from plot_norm.read_norms_new import read_attn
     a = read_attn("logs/run/norms", 0, 40)
     print(a.q_act.shape)   # (S, A, R, B, L, T, H_q)

3. Read both modalities over the same range::

     from plot_norm.read_norms_new import read_both
     h, a = read_both("logs/run/norms", 0, 40)

Running this file directly prints a summary for a given path::

     python -m plot_norm.read_norms_new --norms-dir logs/.../norms --start 0 --end 40
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Return containers
# ---------------------------------------------------------------------------

@dataclass
class HiddenNorms:
    """Dequantized multi-stream hidden norms over a global-step range.

    Shapes use symbols (S, A, R, B, HS, L, T) described in the module docstring.
    ``global_steps`` is the authoritative axis-0 -> global_step mapping; it may be
    sparse (e.g. ``[0, 20, 40]``). ``stream_legend`` names the HS axis; use
    :meth:`stream` to pull one stream as a (S, A, R, B, L, T) array.
    """
    act: np.ndarray            # (S, A, R, B, HS, L, T) float32   activation norms
    grad: np.ndarray           # (S, A, R, B, HS, L, T) float32   gradient norms
    global_steps: np.ndarray   # (S,) int32                       step id per axis-0 slice
    layer_types: List[str]     # length L
    stream_legend: List[str]   # length HS, names the stream axis
    n_elements: int            # embedding dim C
    seq_len: int               # T
    device_batch_size: int     # B
    world_size: int            # R
    grad_accum_steps: int      # A
    outlier_pct: float         # fraction of values kept losslessly per sample

    @property
    def stream_index(self) -> Dict[str, int]:
        """Map stream name -> index into the HS axis."""
        return {name: i for i, name in enumerate(self.stream_legend)}

    def stream(self, name: str, which: str = "act") -> np.ndarray:
        """Return a single stream as a (S, A, R, B, L, T) array.

        ``which`` selects ``"act"`` (default) or ``"grad"``. This is the exact
        shape the legacy 5-D-plus-token plotting code expects, so a single stream
        can be dropped into existing ``plot_norms`` flows unchanged.
        """
        idx = self.stream_index
        if name not in idx:
            raise KeyError(f"unknown stream {name!r}; valid: {list(self.stream_legend)}")
        arr = self.act if which == "act" else self.grad if which == "grad" else None
        if arr is None:
            raise ValueError(f"which must be 'act' or 'grad', got {which!r}")
        return arr[:, :, :, :, idx[name], :, :]


@dataclass
class AttnNorms:
    """Dequantized Q / K / V per-head norms over a global-step range.

    Identical to the old ``read_norms.AttnNorms``. Q has head axis size H_q; K and
    V share head axis size H_kv. ``global_steps`` is the axis-0 -> global_step map.
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
# Internal helpers (carried over from read_norms.py)
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

    Returns fp32 array with shape equal to q.shape. This is generic over the number
    of sample/data axes, so it handles the extra HS axis transparently.
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


def _strided_keep(gs: np.ndarray, step_start: int, step_end: int,
                  skip_every: int, running: int) -> Tuple[np.ndarray, int]:
    """Select which recorded steps of one file to keep, with global striding.

    Of the recorded ``global_steps`` ``gs`` in ``[step_start, step_end]``, keep only
    every ``skip_every``-th one. Striding is applied globally across files via
    ``running`` (in-range steps consumed so far). The first in-range step is always
    kept. ``skip_every == 1`` keeps every in-range step.
    """
    keep = (gs >= step_start) & (gs <= step_end)
    in_range_local = np.where(keep)[0]
    n_in_range = int(in_range_local.size)
    global_idx = running + np.arange(n_in_range)
    sel = in_range_local[(global_idx % skip_every) == 0]
    return sel, n_in_range


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------

def read_hidden(norms_dir: str, step_start: int, step_end: int,
                skip_every: int = 1) -> HiddenNorms:
    """Read multi-stream hidden activation / gradient norms for [start, end].

    Parameters
    ----------
    norms_dir  : path to the ``<logs_dir>/norms`` directory.
    step_start : inclusive lower bound on global_step.
    step_end   : inclusive upper bound on global_step.
    skip_every : positive int (default 1). Of the recorded steps in range, keep only
        every ``skip_every``-th one (globally across files; first in-range step always
        kept). Multiplies the on-disk ``record_every_k_steps`` stride.

    Returns
    -------
    HiddenNorms with ``.act`` / ``.grad`` of shape ``(S, A, R, B, HS, L, T)``.
    ``S`` = number of recorded steps in range surviving the ``skip_every`` stride.
    Consult ``.global_steps`` for the actual ids and ``.stream_legend`` for the HS
    axis; use ``.stream(name)`` to pull one stream as ``(S, A, R, B, L, T)``.

    Raises a clear error if pointed at the *legacy* single-stream format (no
    ``stream_legend`` key) — use ``read_norms.read_hidden`` for those files.
    """
    if skip_every < 1:
        raise ValueError(f"skip_every must be >= 1, got {skip_every}")
    files = _discover_files(norms_dir, "hidden", step_start, step_end)

    act_parts, grad_parts, steps_parts = [], [], []
    meta = None
    running = 0
    for _f, _l, path in files:
        d = np.load(path)
        if "stream_legend" not in d.files:
            raise ValueError(
                f"{path} is the legacy single-stream hidden format (no 'stream_legend'). "
                f"Use plot_norm.read_norms.read_hidden for these files.")
        gs = d["global_steps"]
        sel, n_in_range = _strided_keep(gs, step_start, step_end, skip_every, running)
        running += n_in_range
        if sel.size == 0:
            continue

        act = _dequantize(d["act_q"], d["act_scale"], d["act_min"],
                          d["act_outlier_idx"], d["act_outlier_val"])
        grd = _dequantize(d["grad_q"], d["grad_scale"], d["grad_min"],
                          d["grad_outlier_idx"], d["grad_outlier_val"])
        act_parts.append(act[sel])
        grad_parts.append(grd[sel])
        steps_parts.append(gs[sel])

        if meta is None:
            meta = dict(
                layer_types=_decode_layer_types(d),
                stream_legend=[str(s) for s in d["stream_legend"]],
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


def read_attn(norms_dir: str, step_start: int, step_end: int,
              skip_every: int = 1) -> AttnNorms:
    """Read Q/K/V per-head activation / gradient norms for [start, end].

    Identical behaviour and format to ``read_norms.read_attn`` — the attention dumps
    are unchanged by the multi-stream hidden work. ``skip_every`` behaves as in
    :func:`read_hidden`.

    Returns
    -------
    AttnNorms with ``q_act`` / ``q_grad`` of shape (S, A, R, B, L, T, H_q) and
    ``k_act`` / ``k_grad`` / ``v_act`` / ``v_grad`` of shape (S, A, R, B, L, T, H_kv).
    """
    if skip_every < 1:
        raise ValueError(f"skip_every must be >= 1, got {skip_every}")
    files = _discover_files(norms_dir, "attn", step_start, step_end)

    q_act_p, q_grad_p = [], []
    k_act_p, k_grad_p = [], []
    v_act_p, v_grad_p = [], []
    steps_parts = []
    meta = None
    running = 0
    for _f, _l, path in files:
        d = np.load(path)
        gs = d["global_steps"]
        sel, n_in_range = _strided_keep(gs, step_start, step_end, skip_every, running)
        running += n_in_range
        if sel.size == 0:
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

        q_act_p.append(q_act[sel]);   q_grad_p.append(q_grad[sel])
        k_act_p.append(k_act[sel]);   k_grad_p.append(k_grad[sel])
        v_act_p.append(v_act[sel]);   v_grad_p.append(v_grad[sel])
        steps_parts.append(gs[sel])

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


def read_both(norms_dir: str, step_start: int, step_end: int,
              skip_every: int = 1) -> Tuple[HiddenNorms, AttnNorms]:
    """Convenience: read both modalities for the same step range (aligned steps)."""
    return (read_hidden(norms_dir, step_start, step_end, skip_every),
            read_attn(norms_dir, step_start, step_end, skip_every))


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _print_summary(norms_dir: str, start: int, end: int, skip_every: int = 1) -> None:
    h, a = read_both(norms_dir, start, end, skip_every)
    print(f"[hidden] steps={h.global_steps.tolist()}  act={h.act.shape}  grad={h.grad.shape}")
    print(f"         stream_legend={h.stream_legend}")
    print(f"         layer_types={h.layer_types}")
    print(f"         T={h.seq_len} B={h.device_batch_size} R={h.world_size} "
          f"A={h.grad_accum_steps} C={h.n_elements} outlier_pct={h.outlier_pct}")
    for name in h.stream_legend:
        s_act = h.stream(name, "act"); s_grd = h.stream(name, "grad")
        print(f"         [{name:9s}] act.mean={s_act.mean():.4f}  grad.mean={s_grd.mean():.4e}")
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
    ap.add_argument("--skip-every", type=int, default=1,
                    help="keep every k-th recorded step in range (default 1 = all)")
    args = ap.parse_args()
    _print_summary(args.norms_dir, args.start, args.end, args.skip_every)
