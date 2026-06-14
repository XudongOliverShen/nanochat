"""Reader for the exp3 attention-weight + theory-input dumps (`{step}_attnstat.npz`).

These files are written by the GradientBiasMonitor inside
``scripts_dev/exp3_base_eval.py`` during an eval pass (forward+backward, no optimizer
step) over a fixed, trained model whose attention forward has been monkey-patched to an
eager softmax that materializes the per-head attention weights ``A``.

Each ``{step}_attnstat.npz`` holds everything needed to evaluate the closed-form
position-bias gradient theory (``simulation/gradient_position_bias_simulation.ipynb``):

  * ``A``        : realized causal attention weights, per head — fp16, lower-tri packed.
  * ``mu_x``     : mean of the attention input ``x = norm(block_input)`` per (step, layer).
  * ``Sigma_x``  : covariance of ``x``                          per (step, layer).
  * ``mu_g``     : mean of the upstream per-head output grad ``g = dL/dy`` per (step,layer,head).
  * ``Sigma_g``  : covariance of ``g``                          per (step, layer, head).
  * ``W_q/W_k/W_v`` : projection weights                        per (step, layer).

The EXPERIMENTAL gradient norms ``||dL/dq||²,||dL/dk||²,||dL/dv||²`` live in the sibling
``{step}_attn.npz`` and are read with ``plot_norm.read_norms_new.read_attn``.

═══════════════════════════════════════════════════════════════════════════════════════
ATTENTION-WEIGHT STORAGE FORMAT  (fp16 + lower-triangular packing)
═══════════════════════════════════════════════════════════════════════════════════════
Causal attention is lower-triangular: ``A[...,s,r]=0`` for ``r>s`` (query ``s`` attends
to keys ``r<=s`` only). The monitor stores ONLY the lower triangle (incl. diagonal):

    P = T*(T+1)//2   kept entries per (head, sample)

packed row-major over ``torch.tril_indices(T, T)`` (== ``np.tril_indices(T)``) — i.e. in
the order ``(s=0,r=0),(s=1,r=0),(s=1,r=1),(s=2,r=0),...``. Each value is stored as fp16
(2 bytes). fp16 is used rather than fp8 because at long context the near-uniform weights
are ~1/T (e.g. 1/2048 ~ 5e-4), below the smallest fp8-e4m3 magnitude (~2e-3) -> fp8 would
flush them to 0 and rows would not sum to 1. fp16 reaches ~6e-8, preserving the full row.
To reconstruct dense ``A`` we scatter back via the same ``tril_indices``. ``unpack_attn``
does this.

Array shapes (symbols match read_norms / read_norms_new):
    S=#recorded steps, A=grad_accum, R=world_size, B=device_batch_size,
    H=n_head (query heads), D=head_dim, C=n_embd, T=seq_len, P=T*(T+1)//2.

    attn_weights : (S, A, R, B, L, H, P) float16 (unpack -> (S,A,R,B,L,H,T,T) f32)
    mu_x         : (S, L, C)          f32
    Sigma_x      : (S, L, C, C)       f32
    mu_g         : (S, L, H, D)       f32
    Sigma_g      : (S, L, H, D, D)    f32
    W_q          : (S, L, H*D, C)     f16   (W_k/W_v: H_kv*D rows)

Usage::

    from plot_norm.read_attnstat import read_attnstat, verify_against_theory
    from plot_norm.read_norms_new import read_attn
    s = read_attnstat("logs/run/norms", 0, 4)
    A = s.unpack_attn()                  # (S,A,R,B,L,H,T,T) float32 dense causal weights
    a = read_attn("logs/run/norms", 0, 4)
    verify_against_theory(s, a, step=0, layer=0, head=0)   # prints formula vs measured
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


_FNAME_RE = re.compile(r"step_(\d+)-(\d+)_attnstat\.npz$")


def _discover_files(norms_dir: str, step_start: int, step_end: int) -> List[Tuple[int, int, str]]:
    """Return sorted [(first, last, path), ...] for attnstat files overlapping the range."""
    if step_start > step_end:
        raise ValueError(f"step_start ({step_start}) > step_end ({step_end})")
    out = []
    for path in sorted(glob.glob(os.path.join(norms_dir, "step_*_attnstat.npz"))):
        m = _FNAME_RE.search(os.path.basename(path))
        if m is None:
            continue
        f, l = int(m.group(1)), int(m.group(2))
        if l < step_start or f > step_end:
            continue
        out.append((f, l, path))
    if not out:
        raise FileNotFoundError(
            f"no attnstat files overlap [{step_start},{step_end}] in {norms_dir}")
    out.sort(key=lambda t: t[0])
    return out


def unpack_attn(attn_weights_f16: np.ndarray, seq_len: int) -> np.ndarray:
    """Reconstruct dense causal attention weights from lower-triangular fp16.

    attn_weights_f16 : (..., P) float16, P = T*(T+1)//2, holding the lower triangle in
                       the row-major order of torch.tril_indices(T, T).
    Returns           : (..., T, T) float32 dense array; strict upper triangle is 0.
    """
    T = int(seq_len)
    P = T * (T + 1) // 2
    assert attn_weights_f16.shape[-1] == P, \
        f"packed last dim {attn_weights_f16.shape[-1]} != T*(T+1)/2 = {P}"
    idx = np.tril_indices(T)                                          # (row s, col r), row-major
    vals = np.ascontiguousarray(attn_weights_f16).astype(np.float32)  # (..., P)
    dense = np.zeros(vals.shape[:-1] + (T, T), dtype=np.float32)
    dense[..., idx[0], idx[1]] = vals                                # scatter lower triangle
    return dense


@dataclass
class AttnStat:
    """Dequantized attention weights + theory-input moments over a global-step range."""
    attn_weights: np.ndarray   # (S, A, R, B, L, H, P) float16 (use unpack_attn / .unpack_attn())
    mu_x: np.ndarray           # (S, L, C)
    Sigma_x: np.ndarray        # (S, L, C, C)
    mu_g: np.ndarray           # (S, L, H, D)
    Sigma_g: np.ndarray        # (S, L, H, D, D)
    W_q: np.ndarray            # (S, L, H*D, C)
    W_k: np.ndarray            # (S, L, H_kv*D, C)
    W_v: np.ndarray            # (S, L, H_kv*D, C)
    global_steps: np.ndarray   # (S,) int32
    layer_types: List[str]
    n_head: int
    n_kv_head: int
    head_dim: int
    n_embd: int
    seq_len: int
    device_batch_size: int
    world_size: int
    grad_accum_steps: int

    def unpack_attn(self) -> np.ndarray:
        """Dense causal attention weights (S, A, R, B, L, H, T, T) float32."""
        return unpack_attn(self.attn_weights, self.seq_len)


def _decode_layer_types(d) -> List[str]:
    legend = [str(s) for s in d["layer_type_legend"]]
    return [legend[i] for i in d["layer_types"].tolist()]


def read_attnstat(norms_dir: str, step_start: int, step_end: int) -> AttnStat:
    """Read `{step}_attnstat.npz` files covering [step_start, step_end] (inclusive)."""
    files = _discover_files(norms_dir, step_start, step_end)
    parts = {k: [] for k in ("attn_weights", "mu_x", "Sigma_x", "mu_g", "Sigma_g",
                             "W_q", "W_k", "W_v", "global_steps")}
    meta = None
    for _f, _l, path in files:
        d = np.load(path, allow_pickle=False)
        gs = d["global_steps"]
        keep = (gs >= step_start) & (gs <= step_end)
        sel = np.where(keep)[0]
        if sel.size == 0:
            continue
        for k in parts:
            parts[k].append(d[k][sel] if k != "global_steps" else gs[sel])
        if meta is None:
            meta = dict(
                layer_types=_decode_layer_types(d),
                n_head=int(d["n_head"]),
                n_kv_head=int(d["n_kv_head"]),
                head_dim=int(d["head_dim"]),
                n_embd=int(d["n_embd"]),
                seq_len=int(d["seq_len"]),
                device_batch_size=int(d["device_batch_size"]),
                world_size=int(d["world_size"]),
                grad_accum_steps=int(d["grad_accum_steps"]),
            )
    if meta is None:
        raise FileNotFoundError(
            f"no attnstat records for global_step in [{step_start},{step_end}]")
    cat = lambda key: np.concatenate(parts[key], axis=0)
    return AttnStat(
        attn_weights=cat("attn_weights"),
        mu_x=cat("mu_x"), Sigma_x=cat("Sigma_x"),
        mu_g=cat("mu_g"), Sigma_g=cat("Sigma_g"),
        W_q=cat("W_q"), W_k=cat("W_k"), W_v=cat("W_v"),
        global_steps=cat("global_steps").astype(np.int32),
        **meta,
    )


# ---------------------------------------------------------------------------
# Theory formulas (ported from simulation/gradient_position_bias_simulation.ipynb)
# ---------------------------------------------------------------------------
# These take a realized attention matrix A (single head, (T,T) with A[s,r]=alpha_{r->s})
# and the per-head distribution constants, and predict E||dL/dq_i||^2 etc. per position.

def compute_constants(mu_x, Sigma_x, mu_g, Sigma_g, W_q, W_k, W_v):
    """Build the 7 formula coefficients + value-formula scalars for ONE head.

    Shapes: mu_x (C,), Sigma_x (C,C), mu_g (D,), Sigma_g (D,D),
            W_q/W_k/W_v (D, C).  Mirrors compute_constants() in the notebook.
    """
    Sx = Sigma_x
    d = W_q.shape[0]
    Sigma_q = W_q @ Sx @ W_q.T
    Sigma_v = W_v @ Sx @ W_v.T
    G = np.outer(mu_g, mu_g) + Sigma_g
    C_qv = W_q @ Sx @ W_v.T
    C_vk = W_v @ Sx @ W_k.T
    mu_q = W_q @ mu_x

    sigma_k2   = np.trace(W_k @ Sx @ W_k.T)
    sigma_g2   = np.trace(Sigma_g)
    M_q        = mu_q @ mu_q + np.trace(Sigma_q)
    norm_mu_g2 = mu_g @ mu_g
    norm_mu_q2 = mu_q @ mu_q
    sigma_vg2  = mu_g @ Sigma_v @ mu_g

    tr_G_Sv        = np.trace(G @ Sigma_v)
    tr_Cqv_G_Cqv   = np.trace(C_qv @ G @ C_qv.T)
    tr_Cvk_G_Cvk   = np.trace(C_vk.T @ G @ C_vk)
    norm_Cqv_mu_g2 = (C_qv @ mu_g) @ (C_qv @ mu_g)

    return dict(
        d=d, sigma_g2=sigma_g2, norm_mu_g2=norm_mu_g2,
        C_k_diag      = M_q * tr_G_Sv / d,
        C_k_diag_xtra = 2.0 * tr_Cqv_G_Cqv / d,
        C_k_off       = sigma_vg2 * norm_mu_q2 / d,
        C_k_off_xtra  = norm_Cqv_mu_g2 / d,
        C_q_diag      = sigma_k2 * tr_G_Sv / d,
        C_q_diag_xtra = 2.0 * tr_Cvk_G_Cvk / d,
        C_q_off       = tr_Cvk_G_Cvk / d,
    )


def _lambda_matrix(A):
    return A @ A.T   # (T,T): Lambda[s,s'] = sum_r alpha_{r->s} alpha_{r->s'}


def value_formula(A, c):
    """E||dL/dv_i||^2 per key position i. Exact. A: (T,T), A[s,r]=alpha_{r->i==r}."""
    col_sum   = A.sum(axis=0)          # sum over query rows s -> (T,) indexed by key i
    col_sqsum = (A ** 2).sum(axis=0)
    return c["norm_mu_g2"] * col_sum ** 2 + c["sigma_g2"] * col_sqsum


def key_formula(A, c):
    """E||dL/dk_i||^2 per key position i (4-term approximation)."""
    T = A.shape[0]
    Lam = _lambda_matrix(A)
    diagLam = np.diagonal(Lam)
    alpha_ss = np.diagonal(A)
    out = np.zeros(T)
    for i in range(T):
        a = A[:, i]
        beta = -alpha_ss.copy()
        beta[i] = 1.0 - alpha_ss[i]
        sum_a = a.sum()
        sum_a2 = (a ** 2).sum()
        t_diag = (a ** 2 * (1.0 - 2.0 * a + diagLam)).sum()
        t_diag_xtra = (a ** 2 * beta ** 2).sum()
        full1 = sum_a ** 2 - 2.0 * sum_a2 * sum_a + np.einsum("s,sr,r->", a, Lam, a)
        t_off = full1 - t_diag
        ab = a * beta
        t_off_xtra = ab.sum() ** 2 - (ab ** 2).sum()
        out[i] = (c["C_k_diag"] * t_diag + c["C_k_diag_xtra"] * t_diag_xtra
                  + c["C_k_off"] * t_off + c["C_k_off_xtra"] * t_off_xtra)
    return out


def query_formula(A, c):
    """E||dL/dq_i||^2 per query position i (3-term approximation)."""
    T = A.shape[0]
    diagLam = np.diagonal(_lambda_matrix(A))
    out = np.zeros(T)
    for i in range(T):
        b = A[i, :]
        Lam_i = diagLam[i]
        sum_b2 = (b ** 2).sum()
        t_diag = (b ** 2 * (1.0 - 2.0 * b + Lam_i)).sum()
        t_diag_xtra = (b ** 2 * (1.0 - b) ** 2).sum()
        full = 1.0 - 2.0 * sum_b2 + 2.0 * sum_b2 ** 2
        diag = (b ** 2 * (1.0 - 2.0 * b + 2.0 * b ** 2)).sum()
        t_off = full - diag
        out[i] = c["C_q_diag"] * t_diag + c["C_q_diag_xtra"] * t_diag_xtra + c["C_q_off"] * t_off
    return out


def verify_against_theory(stat: AttnStat, attn, step: int = 0, layer: int = 0, head: int = 0,
                          accum: int = 0, rank: int = 0, batch: int = 0):
    """Compare theory formulas (evaluated on the recorded A + moments + W) against the
    measured per-position gradient norms from read_norms_new.read_attn.

    `attn` is the AttnNorms object from `read_norms_new.read_attn` over the same range.
    Prints per-gradient mean |%error| (value gradient is the exact formula -> should match
    to MC/quantization noise; key/query carry the fixed-attention approximation error).

    NOTE on indexing: the recorded `A` (B,H,T,T) has A[...,s,r] = softmax over keys r for
    query s, i.e. A[s,r] = alpha_{r->s} — exactly the notebook convention.
    """
    s = int(np.where(stat.global_steps == stat.global_steps[step])[0][0]) if step >= len(stat.global_steps) else step
    D, C = stat.head_dim, stat.n_embd
    H = stat.n_head

    # Per-head projection weight slices (W_* rows are head-major: head h -> rows h*D:(h+1)*D).
    Wq = stat.W_q[s, layer].astype(np.float64).reshape(H, D, C)[head]
    # K/V may have fewer heads (GQA); map query head -> kv head.
    n_kv = stat.n_kv_head
    kv_head = head // (H // n_kv) if n_kv > 0 else 0
    Wk = stat.W_k[s, layer].astype(np.float64).reshape(n_kv, D, C)[kv_head]
    Wv = stat.W_v[s, layer].astype(np.float64).reshape(n_kv, D, C)[kv_head]

    mu_x = stat.mu_x[s, layer].astype(np.float64)
    Sigma_x = stat.Sigma_x[s, layer].astype(np.float64)
    mu_g = stat.mu_g[s, layer, head].astype(np.float64)
    Sigma_g = stat.Sigma_g[s, layer, head].astype(np.float64)

    consts = compute_constants(mu_x, Sigma_x, mu_g, Sigma_g, Wq, Wk, Wv)

    A = stat.unpack_attn()[s, accum, rank, batch, layer, head].astype(np.float64)  # (T,T)

    pred = {
        "v": value_formula(A, consts),
        "k": key_formula(A, consts),
        "q": query_formula(A, consts),
    }
    # Measured: ||grad||^2 per position, averaged over (accum,rank,batch) at this step/layer/head.
    meas = {
        "q": (attn.q_grad[s, :, :, :, layer, :, head] ** 2).mean(axis=(0, 1, 2)),
        "k": (attn.k_grad[s, :, :, :, layer, :, kv_head] ** 2).mean(axis=(0, 1, 2)),
        "v": (attn.v_grad[s, :, :, :, layer, :, kv_head] ** 2).mean(axis=(0, 1, 2)),
    }
    print(f"verify_against_theory: step_idx={s} (global_step={int(stat.global_steps[s])}), "
          f"layer={layer}, head={head}")
    for kk in ("v", "k", "q"):
        p, m = pred[kk], meas[kk]
        denom = np.where(np.abs(m) < 1e-12, 1.0, m)
        pe = 100.0 * (p - m) / denom
        tag = "exact" if kk == "v" else "approx"
        print(f"  {kk} ({tag:6s}): mean|%err|={np.abs(pe).mean():9.3f}  "
              f"median|%err|={np.median(np.abs(pe)):9.3f}  "
              f"pred.mean={p.mean():.4e}  meas.mean={m.mean():.4e}")
    return dict(pred=pred, meas=meas, consts=consts, A=A)
