"""
exp3 eval-time recorder for the position-bias attention-gradient theory.

This is a FORK of scripts_dev/exp2_base_train.py used as an *eval* pass (forward+backward,
NO optimizer step — the model is loaded from a checkpoint and stays fixed). It reuses all of
exp2's GradientBiasMonitor machinery (4-stream hidden norms + per-head Q/K/V act/grad norms,
large-file chunking, flush-every-K, DDP gather) and ADDS recording of the quantities needed to
verify the closed-form gradient theory, written to a new `{step}_attnstat.npz` file.

Run as (single GPU; disable torch.compile so the eager attention + hooks are reliable):

    TORCHDYNAMO_DISABLE=1 torchrun --standalone --nproc_per_node=1 \
        -m scripts_dev.exp3_base_eval -- --model-tag=<tag> [--step=<n>] \
        --depth=... --max-seq-len=... --no-rope --no-qknorm --no-qk-scale ...

═══════════════════════════════════════════════════════════════════════════════════════════
THEORY  (see simulation/gradient_position_bias_simulation.ipynb)
═══════════════════════════════════════════════════════════════════════════════════════════
Single-head causal scaled-dot-product attention with i.i.d. Gaussian inputs and an upstream
gradient sampled independently of the inputs:

    x_i ~ N(mu_x, Sigma_x)                         # input tokens (here: the attn input norm(x))
    q_i = W_Q x_i,  k_i = W_K x_i,  v_i = W_V x_i  # LINEAR projections (no RoPE/QK-norm/scale)
    alpha_{r->s} = softmax_r( q_s · k_r / sqrt(d) ) over r <= s    # causal attention weights A
    y_s = sum_{r<=s} alpha_{r->s} v_r              # attention output
    g_s = dL/dy_s ~ N(mu_g, Sigma_g),  independent of x            # upstream gradient

The theory predicts the EXPECTED squared per-position gradient norms E||dL/dq_i||^2,
E||dL/dk_i||^2, E||dL/dv_i||^2 as closed forms in terms of:

  * the realized attention matrix A         -> recorded here (fp8, lower-tri packed)
  * input moments   mu_x (C,), Sigma_x (C,C) -> recorded here (per layer, per recorded step)
  * grad moments    mu_g (D,), Sigma_g (D,D) -> recorded here (per layer, per head, per step)
  * projection weights W_Q, W_K, W_V         -> recorded here (per layer, per step)

From these, compute_constants() in the notebook builds:
    Sigma_q = W_Q Sigma_x W_Q^T,  Sigma_v = W_V Sigma_x W_V^T,  G = mu_g mu_g^T + Sigma_g,
    C_qv = W_Q Sigma_x W_V^T,     C_vk = W_V Sigma_x W_K^T,      mu_q = W_Q mu_x,  etc.

Formulas under test:
  * VALUE (exact):
        E||dL/dv_i||^2 = ||mu_g||^2 (sum_{s>=i} alpha_{i->s})^2 + sigma_g^2 sum_{s>=i} alpha_{i->s}^2
  * KEY / QUERY (approximate): the 4-term / 3-term boxed results with the same-position
    C_qv / C_vk fourth-order corrections. Valid under the "fixed-attention" approximation
    (treat A as decoupled from q,k,v fluctuations): exact for near-uniform attention,
    degrading as attention sharpens.

EXPERIMENTAL side (already recorded by the monitor in `{step}_attn.npz`): the measured
per-head per-position ||dL/dq_i||^2, ||dL/dk_i||^2, ||dL/dv_i||^2 (q_grad/k_grad/v_grad).
verify_against_theory() in plot_norm/read_attnstat.py compares formula(A, moments, W) to these.

CAVEAT: the theory assumes the LINEAR map q = W_Q x with NO RoPE / QK-norm / x1.2 sharpening.
For a faithful test, run this eval with --no-rope --no-qknorm --no-qk-scale (the model was
trained with those off). Because we record the realized A directly, any formula evaluated on
that A stays self-consistent regardless of these knobs.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# =============================================================================
# === NEW (norm monitoring): GradientBiasMonitor — collects per-token hidden
# === and Q/K/V activation + gradient norms via forward/backward hooks on the
# === uncompiled model; flushes dense-matrix npz chunks to <logs_dir>/norms.
# === See class docstring for file/array layout.
# =============================================================================
import numpy as np
from datetime import datetime


def _quantize_uint8_batched(arr, outlier_pct):
    """Vectorized per-row uint8 quantization with top-`outlier_pct` preserved losslessly.

    arr: float32 (N, M). Returns:
      q            uint8   (N, M)
      scale        float32 (N,)
      mn           float32 (N,)
      outlier_idx  int32   (N, K)   K = max(1, int(M * outlier_pct))
      outlier_val  float32 (N, K)

    Dequant (row i): val = mn[i] + q[i] * scale[i], then splice
    outlier_val[i] into positions outlier_idx[i].
    """
    assert arr.ndim == 2 and arr.dtype == np.float32
    N, M = arr.shape
    K = max(1, int(M * outlier_pct)) if outlier_pct > 0 else 1

    top_idx = np.argpartition(arr, -K, axis=1)[:, -K:].astype(np.int32)          # (N, K)
    outlier_val = np.take_along_axis(arr, top_idx, axis=1).astype(np.float32)    # (N, K)

    # replace outlier positions with per-row min of the original row, then
    # compute quant params on the cleaned array (mirrors the original scalar impl).
    row_min_orig = arr.min(axis=1, keepdims=True).astype(np.float32)             # (N, 1)
    a_clean = arr.copy()
    rows = np.arange(N)[:, None]
    a_clean[rows, top_idx] = row_min_orig

    mn = a_clean.min(axis=1).astype(np.float32)                                  # (N,)
    mx = a_clean.max(axis=1).astype(np.float32)
    rng = mx - mn
    scale = np.where(rng > 0, rng / 255.0, 1.0).astype(np.float32)               # (N,)
    q = np.clip(np.round((a_clean - mn[:, None]) / scale[:, None]), 0, 255).astype(np.uint8)
    return q, scale, mn, top_idx, outlier_val


# =============================================================================
# exp3: lower-triangular fp16 packing of causal attention weights A
# =============================================================================
# Causal attention is lower-triangular: A[..., s, r] = 0 for r > s (a query at
# position s only attends to keys r <= s). We therefore store ONLY the lower
# triangle (including diagonal), which is exactly half (+ diagonal) the entries:
#   number of kept entries = T*(T+1)/2.
#
# Packing layout (row-major over the lower triangle):
#   idx = torch.tril_indices(T, T)  ->  shape (2, T*(T+1)/2)
#   idx[0] = row (query) positions s, idx[1] = col (key) positions r, with r <= s,
#   ordered as (s=0,r=0), (s=1,r=0), (s=1,r=1), (s=2,r=0), ... (row-major).
#   packed[..., m] = A[..., idx[0][m], idx[1][m]]
#
# Quantization: fp16 (float16), 2 bytes per value. We use fp16 rather than fp8
# because at long context the near-uniform attention weights are ~1/T (e.g.
# 1/2048 ~ 5e-4), BELOW the smallest fp8-e4m3 magnitude (~2e-3) -> they would
# flush to 0 and rows would not sum to 1. fp16 represents values down to ~6e-8,
# so the full attention row is preserved and row-sums stay ~1.0.
#
# To UNPACK (in the standalone reader):
#   import torch
#   idx = torch.tril_indices(T, T)
#   vals = packed_f16.astype(np.float32)                  # (..., T*(T+1)/2)
#   A = np.zeros(packed_f16.shape[:-1] + (T, T), np.float32)
#   A[..., idx[0], idx[1]] = vals          # scatter lower triangle back; upper stays 0
#
# A_PACKED_LEN(T) == T*(T+1)//2.
def _tril_pack_f16(A_dense):
    """Pack a dense causal attention tensor (..., T, T) into lower-triangular fp16.

    Returns a float16 numpy array of shape (..., T*(T+1)//2) holding the lower
    triangle in row-major order (see module comment for the exact layout).
    Input may be a torch tensor (any device) or numpy array.
    """
    if not torch.is_tensor(A_dense):
        A_dense = torch.as_tensor(A_dense)
    T = A_dense.shape[-1]
    idx = torch.tril_indices(T, T, device=A_dense.device)            # (2, T*(T+1)/2)
    packed = A_dense[..., idx[0], idx[1]]                            # (..., T*(T+1)/2)
    return packed.to(torch.float16).cpu().numpy()


class GradientBiasMonitor:
    """Capture per-token hidden / Q / K / V norms (fwd activations + bwd grads).

    Hooks are attached on the uncompiled model (`orig_model.transformer.h[i]`).
    For each layer, four per-token hidden residual-stream norms are recorded
    (block_in, attn_in, attn_out, block_out; see stream_legend). Q/K/V
    projections are recorded per-head (no head-mixing). `flush(step)` gathers
    records across DDP ranks (all_gather_object) and writes dense-matrix npz
    chunks on rank 0.

    ────────────────────────────────────────────────────────────────────────────
    OUTPUT FILES (under `<logs_dir>/norms/`, one pair per flush window)

        step_{first}-{last}_hidden.npz
        step_{first}-{last}_attn.npz

    Flush windows cover `steps_per_file` consecutive global_steps (final flush
    may be shorter). Within a window, dimension symbols are:

        S = number of global_steps covered       A = grad_accum_steps
        R = world_size (# DDP ranks)             B = device_batch_size
        L = n_layer                              T = seq_len
        H_q  = n_head (query heads)              H_kv = n_kv_head (kv heads; GQA)

    Leading-axis convention (all stacked arrays): (S, A, R, B, L, …data…).

    ─── {step_tag}_hidden.npz ───────────────────────────────────────────────
    HS = 4 per-token residual-stream points per layer, indexed by stream_legend
    (axis sits between B and L):
      ["block_in", "attn_in", "attn_out", "block_out"]
    For a block `x = x + attn(norm(x)); x = x + mlp(norm(x))`:
      block_in  = x (block input)            attn_in  = norm(x) (attn input)
      attn_out  = attn(norm(x)) (attn out)   block_out = block output

      act_q              uint8   (S, A, R, B, HS, L, T)     quantized activation norms
      act_scale          float32 (S, A, R, B, HS, L)        per-sample scale
      act_min            float32 (S, A, R, B, HS, L)        per-sample min
      act_outlier_idx    int32   (S, A, R, B, HS, L, K_h)   positions in [0, T)
      act_outlier_val    float32 (S, A, R, B, HS, L, K_h)
      grad_q / grad_scale / grad_min / grad_outlier_idx / grad_outlier_val
                         same shapes as act_*               gradient norms
      global_steps       int32   (S,)                       step value of axis-0 slice
      layer_types        uint8   (L,)                       index into layer_type_legend
      layer_type_legend  str     (2,)   ["full_attention","sliding_attention"]
      stream_legend      str     (HS,)  ["block_in","attn_in","attn_out","block_out"]
      format_version     int32   scalar (2 = multi-stream hidden)
      n_elements / seq_len / device_batch_size / world_size / grad_accum_steps
                         int32   scalars                    (n_elements = embed dim C)
      outlier_pct        float32 scalar

      K_h = max(1, int(T * outlier_pct))

    ─── {step_tag}_attn.npz ─────────────────────────────────────────────────
    GQA means H_q ≠ H_kv, so two rectangular groups share one file:
      q_*  : query-only, head axis size H_q
      kv_* : key+value, extra axis after L with size 2 (0=k, 1=v), head size H_kv

      q_act_q            uint8   (S, A, R, B, L, T, H_q)
      q_act_scale        float32 (S, A, R, B, L)
      q_act_min          float32 (S, A, R, B, L)
      q_act_outlier_idx  int32   (S, A, R, B, L, K_q)       flat indices into (T*H_q,)
      q_act_outlier_val  float32 (S, A, R, B, L, K_q)
      q_grad_*           (mirror of q_act_*)

      kv_act_q           uint8   (S, A, R, B, L, 2, T, H_kv)
      kv_act_scale       float32 (S, A, R, B, L, 2)
      kv_act_min         float32 (S, A, R, B, L, 2)
      kv_act_outlier_idx int32   (S, A, R, B, L, 2, K_kv)   flat indices into (T*H_kv,)
      kv_act_outlier_val float32 (S, A, R, B, L, 2, K_kv)
      kv_grad_*          (mirror of kv_act_*)

      global_steps       int32   (S,)
      layer_types        uint8   (L,)
      layer_type_legend  str     (2,)   ["full_attention","sliding_attention"]
      kv_legend          str     (2,)   ["k","v"]   — for axis 5 of kv_*
      n_head / n_kv_head / head_dim / seq_len / device_batch_size / world_size /
        grad_accum_steps int32   scalars
      outlier_pct        float32 scalar

      K_q  = max(1, int(T * H_q  * outlier_pct))
      K_kv = max(1, int(T * H_kv * outlier_pct))

    ─── INDEX RECOVERY ──────────────────────────────────────────────────────
    For hidden array `X[s, a, r, b, hs, l, t]`:
      global_step = global_steps[s]   accum_idx = a   rank = r
      sample_idx  = b                 stream   = stream_legend[hs]
      layer_idx   = l                 layer_type = layer_type_legend[layer_types[l]]

    For attention:
      Q value for sample (s,a,r,b,l) at token t head h: q_act_q[s,a,r,b,l,t,h]
      K value: kv_act_q[s,a,r,b,l,0,t,h]    V value: kv_act_q[s,a,r,b,l,1,t,h]

    ─── DEQUANTIZATION ──────────────────────────────────────────────────────
    Per sample: val = min + q.astype(float32) * scale
    Then overwrite positions in outlier_idx with outlier_val.
    For attention samples, outlier indices are into the C-order flattened
    (T, H) view: `flat_idx = t * H + h`.
    """

    def __init__(self, orig_model, logs_dir, rank=0, world_size=1,
                 steps_per_file=2, outlier_pct=0.01,
                 record_every_k_steps=1, debug=False):
        self.model = orig_model
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._logs_dir = logs_dir
        self._debug = debug
        self._steps_per_file = int(steps_per_file)
        self._outlier_pct = float(outlier_pct)
        self._record_every_k = max(1, int(record_every_k_steps))
        self._first_fwd_reported = False

        self._global_step = 0
        self._accum_idx = 0
        self._s_idx = 0                   # within-window recorded-step index
        self._record_this_step = False    # gate set by set_step()
        self._recorded_steps = []         # global_steps currently in the staging slabs
        self._configured = False          # True once configure() has allocated slabs
        self._hooks = []

        spec = self._introspect_model(orig_model)
        self._hidden_modules    = spec["hidden_modules"]      # list of modules (forward output = hidden state)
        self._attn_specs        = spec["attn_specs"]          # [{layer_idx, proj:{q,k,v}, n_heads:{q,k,v}, head_dim}, ...]
        self._layer_types       = spec["layer_types"]         # list[str] parallel to hidden_modules
        self._layer_type_legend = spec["layer_type_legend"]   # list[str] of distinct categories
        self._n_head            = spec["n_head"]
        self._n_kv_head         = spec["n_kv_head"]
        self._head_dim          = spec["head_dim"]
        self._n_layer           = len(self._hidden_modules)
        self._n_embd            = int(orig_model.config.n_embd)   # C, attn input dim
        # exp3: every attention module (for toggling eager A capture per record step)
        self._attn_modules = [s["attn"] for s in self._attn_specs]

        # precompute metadata arrays used at flush time
        _t2i = {t: i for i, t in enumerate(self._layer_type_legend)}
        self._layer_type_legend_arr = np.array(self._layer_type_legend)
        self._layer_types_arr = np.array([_t2i[t] for t in self._layer_types], dtype=np.uint8)

        if self.rank == 0:
            os.makedirs(os.path.join(logs_dir, "norms"), exist_ok=True)

        self._setup_hooks()

    # ---------------- step/accum tracking ----------------
    def configure(self, grad_accum_steps, device_batch_size, seq_len):
        """Allocate CPU fp16 staging slabs. Call once before the training loop,
        after grad_accum_steps / device_batch_size / seq_len are known."""
        A = int(grad_accum_steps)
        B = int(device_batch_size)
        T = int(seq_len)
        K_win = self._steps_per_file
        L = self._n_layer
        H_q = self._n_head
        H_kv = self._n_kv_head
        self._A, self._B, self._T = A, B, T
        # Hidden slabs: (K_win, A, HS, L, B, T)   HS = # residual-stream points per layer
        HS = len(self._HIDDEN_STREAMS)
        self._hidden_act  = torch.zeros((K_win, A, HS, L, B, T), dtype=torch.float16)
        self._hidden_grad = torch.zeros((K_win, A, HS, L, B, T), dtype=torch.float16)
        # Q slabs: (K_win, A, L, B, T, H_q)
        self._q_act  = torch.zeros((K_win, A, L, B, T, H_q), dtype=torch.float16)
        self._q_grad = torch.zeros((K_win, A, L, B, T, H_q), dtype=torch.float16)
        # KV slabs: (K_win, A, L, 2, B, T, H_kv)   axis-3: 0=k, 1=v
        self._kv_act  = torch.zeros((K_win, A, L, 2, B, T, H_kv), dtype=torch.float16)
        self._kv_grad = torch.zeros((K_win, A, L, 2, B, T, H_kv), dtype=torch.float16)

        # === exp3: attention-weight + moment staging ===
        C = self._n_embd
        D = self._head_dim
        # Attention weights A, per (recorded-step, accum, layer, batch, head), stored as
        # lower-triangular fp8 bytes (uint8). P = T*(T+1)/2 packed entries per (head, sample).
        P = T * (T + 1) // 2
        self._attn_P = P
        # CPU fp16 slab. Dominant storage term; sized H_q (query heads) since A is per query head.
        self._attnw_packed = torch.zeros((K_win, A, L, B, H_q, P), dtype=torch.float16)
        # Input moments of the attention input x = norm(block_input), accumulated over
        # (accum, batch, token positions) per (recorded-step, layer). fp64 running sums.
        self._x_count = torch.zeros((K_win, L), dtype=torch.float64)               # # of x vectors summed
        self._x_sum   = torch.zeros((K_win, L, C), dtype=torch.float64)            # sum_i x_i
        self._x_xxT   = torch.zeros((K_win, L, C, C), dtype=torch.float64)         # sum_i x_i x_i^T
        # Upstream-gradient moments of the per-head attention output gradient g = dL/dy,
        # accumulated over (accum, batch, positions) per (recorded-step, layer, head).
        self._g_count = torch.zeros((K_win, L, H_q), dtype=torch.float64)          # # of g vectors summed
        self._g_sum   = torch.zeros((K_win, L, H_q, D), dtype=torch.float64)       # sum_i g_i
        self._g_ggT   = torch.zeros((K_win, L, H_q, D, D), dtype=torch.float64)    # sum_i g_i g_i^T
        self._configured = True

    def set_step(self, global_step):
        self._global_step = int(global_step)
        self._accum_idx = 0
        self._record_this_step = (self._global_step % self._record_every_k == 0)
        if self._record_this_step:
            self._s_idx = len(self._recorded_steps)
            self._recorded_steps.append(self._global_step)
        # exp3: only let the eager attention stash A on record steps (avoids the slab cost
        # and large (T,T) tensors on non-record steps).
        for _attn in self._attn_modules:
            _attn._record_attn_weights = self._record_this_step

    def advance_accum(self):
        self._accum_idx += 1

    # ---------------- model-specific introspection ----------------
    @staticmethod
    def _introspect_model(orig_model):
        """Discover hook points and layer metadata for a given model.

        This is the ONLY model-specific method. To port GradientBiasMonitor to a
        new architecture, edit this method. Everything else is model-agnostic
        and works off the returned spec.

        Returns a dict:
          hidden_modules    : list of nn.Module — forward output is hidden state (B, T, C)
          attn_specs        : list of {layer_idx, proj:{q,k,v}, n_heads:{q,k,v}, head_dim}
                              — one entry per block that has a Q/K/V attention subblock;
                              n_heads may differ across q vs k/v (GQA) but MUST be the same
                              across layers for the current dense-matrix flush layout.
          layer_types       : list[str], one per hidden_module (used for analysis grouping)
          layer_type_legend : list[str] of all distinct category labels that can appear in
                              layer_types (fixed up-front so the legend is stable across
                              checkpoints even if a run happens not to use some label).
          n_head / n_kv_head / head_dim : int scalars shared across attn_specs.
        """
        # --- hidden-state hook points: each transformer block's output (B, T, C) ---
        hidden_modules = list(orig_model.transformer.h)

        # --- layer-type classification (full vs sliding attention) ---
        seq_len = orig_model.config.sequence_len
        layer_types = []
        for i in range(len(hidden_modules)):
            left, _right = orig_model.window_sizes[i]
            if left < 0 or left >= seq_len:
                layer_types.append("full_attention")
            else:
                layer_types.append("sliding_attention")
        layer_type_legend = ["full_attention", "sliding_attention"]

        # --- Q/K/V hook points: auto-detect blocks that expose c_q / c_k / c_v ---
        attn_specs = []
        for i, block in enumerate(hidden_modules):
            attn = getattr(block, "attn", None)
            if attn is None or not all(hasattr(attn, n) for n in ("c_q", "c_k", "c_v")):
                continue
            attn_specs.append({
                "layer_idx": i,
                "attn":    attn,                  # the attention module (exp3: eager A + attn_in moments)
                "c_proj":  getattr(attn, "c_proj", None),  # output proj (exp3: upstream-grad moments hook)
                "proj":    {"q": attn.c_q, "k": attn.c_k, "v": attn.c_v},
                "n_heads": {"q": int(attn.n_head),
                            "k": int(attn.n_kv_head),
                            "v": int(attn.n_kv_head)},
                "head_dim": int(attn.head_dim),
            })

        # current flush layout requires shared head shapes across attention layers
        if attn_specs:
            s0 = attn_specs[0]
            n_head, n_kv_head, head_dim = s0["n_heads"]["q"], s0["n_heads"]["k"], s0["head_dim"]
            for s in attn_specs[1:]:
                assert s["n_heads"]["q"] == n_head and s["n_heads"]["k"] == n_kv_head, \
                    "GradientBiasMonitor requires identical head counts across attn layers"
                assert s["head_dim"] == head_dim, \
                    "GradientBiasMonitor requires identical head_dim across attn layers"
        else:
            n_head = n_kv_head = head_dim = 0

        return {
            "hidden_modules": hidden_modules,
            "attn_specs": attn_specs,
            "layer_types": layer_types,
            "layer_type_legend": layer_type_legend,
            "n_head": n_head, "n_kv_head": n_kv_head, "head_dim": head_dim,
        }

    # ---------------- hook setup ----------------
    _KV_IDX = {"k": 0, "v": 1}
    # Per-token hidden residual-stream points captured per layer (act + grad).
    # For a block `x = x + attn(norm(x)); x = x + mlp(norm(x))`:
    #   block_in  = x (block input)            attn_in  = norm(x) (attn input)
    #   attn_out  = attn(norm(x)) (attn output) block_out = block output
    _HIDDEN_STREAMS = ("block_in", "attn_in", "attn_out", "block_out")
    _HS_IDX = {name: i for i, name in enumerate(_HIDDEN_STREAMS)}

    def _write_hidden(self, slab, stream_name, layer_idx, tensor):
        """Compute per-token L2 norm of a (B, T, C) tensor and copy into the
        right (stream, layer) slot of a hidden slab. No-op if tensor is not 3-D."""
        if tensor is None or tensor.dim() != 3:
            return
        norms = tensor.detach().float().norm(dim=-1).to(torch.float16)  # (B, T) on GPU
        slab[self._s_idx, self._accum_idx, self._HS_IDX[stream_name], layer_idx].copy_(norms)

    # ---------------- exp3: attention weights + theory-input moments ----------------
    def _capture_attn_weights(self, layer_idx, attn_module):
        """Read the eager-stashed per-head attention weights A (B, H_q, T, T), pack the
        causal lower triangle to fp8 bytes, and copy into the uint8 staging slab."""
        A = getattr(attn_module, "_last_attn_weights", None)
        if A is None:
            return
        packed_f16 = _tril_pack_f16(A)                         # numpy (B, H_q, P) float16
        dst = self._attnw_packed[self._s_idx, self._accum_idx, layer_idx]   # (B, H_q, P)
        dst.copy_(torch.from_numpy(packed_f16))

    def _accum_input_moments(self, layer_idx, x):
        """Accumulate sum_i x_i and sum_i x_i x_i^T over all tokens in x (B, T, C),
        into the running moment accumulators for (recorded-step, layer)."""
        if x is None or x.dim() != 3:
            return
        xf = x.detach().float().reshape(-1, x.shape[-1])       # (B*T, C)
        self._x_count[self._s_idx, layer_idx] += xf.shape[0]
        self._x_sum[self._s_idx, layer_idx]   += xf.sum(dim=0).double().cpu()
        self._x_xxT[self._s_idx, layer_idx]   += (xf.transpose(0, 1) @ xf).double().cpu()

    def _accum_grad_moments(self, layer_idx, g_concat):
        """Accumulate per-head sum_i g_i and sum_i g_i g_i^T from the concatenated
        per-head attention-output gradient g_concat (B, T, H_q*D), into the running
        moment accumulators for (recorded-step, layer, head)."""
        if g_concat is None or g_concat.dim() != 3:
            return
        H, D = self._n_head, self._head_dim
        g = g_concat.detach().float().reshape(-1, H, D)        # (B*T, H, D)
        n = g.shape[0]
        self._g_count[self._s_idx, layer_idx] += n
        self._g_sum[self._s_idx, layer_idx]   += g.sum(dim=0).double().cpu()        # (H, D)
        # per-head outer products: sum_i g_i g_i^T  -> (H, D, D)
        ggT = torch.einsum("nhd,nhe->hde", g, g)
        self._g_ggT[self._s_idx, layer_idx]   += ggT.double().cpu()

    def _setup_hooks(self):
        # --- hidden-state hooks: capture 4 residual-stream points per layer ---
        # Each layer needs only two module hook pairs:
        #   * the Block module       -> block_in (fwd inp[0] / bwd grad_input[0])
        #                               block_out (fwd output / bwd grad_output[0])
        #   * the block.attn module  -> attn_in  (fwd inp[0] / bwd grad_input[0])
        #                               attn_out (fwd output / bwd grad_output[0])
        def _first(x):
            return x[0] if isinstance(x, tuple) else x

        for i, block in enumerate(self._hidden_modules):

            # ---- Block: block_in (input) + block_out (output) ----
            def make_fwd_block(layer_idx):
                def fwd_hook(module, inp, output):
                    if not module.training or not self._record_this_step:
                        return
                    if self._debug and not self._first_fwd_reported and layer_idx == 0:
                        h = _first(output)
                        print(f"[monitor] first hidden fwd fired (layer=0, shape={tuple(h.shape)})")
                        self._first_fwd_reported = True
                    self._write_hidden(self._hidden_act, "block_in",  layer_idx, _first(inp))
                    self._write_hidden(self._hidden_act, "block_out", layer_idx, _first(output))
                return fwd_hook

            def make_bwd_block(layer_idx):
                def bwd_hook(module, grad_input, grad_output):
                    if not module.training or not self._record_this_step:
                        return
                    self._write_hidden(self._hidden_grad, "block_in",  layer_idx, _first(grad_input))
                    self._write_hidden(self._hidden_grad, "block_out", layer_idx, _first(grad_output))
                return bwd_hook

            self._hooks.append(block.register_forward_hook(make_fwd_block(i)))
            self._hooks.append(block.register_full_backward_hook(make_bwd_block(i)))

            # ---- attn: attn_in (input = norm(x)) + attn_out (output) ----
            attn = getattr(block, "attn", None)
            if attn is None:
                continue

            def make_fwd_attn(layer_idx):
                def fwd_hook(module, inp, output):
                    if not module.training or not self._record_this_step:
                        return
                    attn_in = _first(inp)   # norm(x): the attention input (B, T, C)
                    self._write_hidden(self._hidden_act, "attn_in",  layer_idx, attn_in)
                    self._write_hidden(self._hidden_act, "attn_out", layer_idx, _first(output))
                    # exp3: capture realized attention weights A (stashed by eager forward),
                    # and accumulate the input moments mu_x / Sigma_x from norm(x).
                    self._capture_attn_weights(layer_idx, module)
                    self._accum_input_moments(layer_idx, attn_in)
                return fwd_hook

            def make_bwd_attn(layer_idx):
                def bwd_hook(module, grad_input, grad_output):
                    if not module.training or not self._record_this_step:
                        return
                    self._write_hidden(self._hidden_grad, "attn_in",  layer_idx, _first(grad_input))
                    self._write_hidden(self._hidden_grad, "attn_out", layer_idx, _first(grad_output))
                return bwd_hook

            self._hooks.append(attn.register_forward_hook(make_fwd_attn(i)))
            self._hooks.append(attn.register_full_backward_hook(make_bwd_attn(i)))

            # ---- exp3: c_proj backward hook -> upstream per-head attention-output grad moments ----
            # c_proj maps the concatenated per-head attention output y (B,T,H*D) -> residual (B,T,C).
            # Its grad_input[0] = dL/d(c_proj input) = dL/dy, the per-head attention-output gradient
            # the theory calls g_s. We accumulate mu_g / Sigma_g per (step, layer, head) from it.
            c_proj = getattr(attn, "c_proj", None)
            if c_proj is not None:
                def make_bwd_cproj(layer_idx):
                    def bwd_hook(module, grad_input, grad_output):
                        if not module.training or not self._record_this_step:
                            return
                        self._accum_grad_moments(layer_idx, _first(grad_input))
                    return bwd_hook
                self._hooks.append(c_proj.register_full_backward_hook(make_bwd_cproj(i)))

        # --- Q/K/V hooks on every discovered attention layer ---
        for spec in self._attn_specs:
            i = spec["layer_idx"]
            head_dim = spec["head_dim"]
            for qkv_name, proj in spec["proj"].items():
                n_heads = spec["n_heads"][qkv_name]

                def make_fwd_qkv(layer_idx, qkv, n_heads, head_dim):
                    is_q = (qkv == "q")
                    kv_i = self._KV_IDX.get(qkv, 0)
                    def fwd_hook(module, inp, output):
                        if not module.training or not self._record_this_step:
                            return
                        # print(f"[fwd]\t{'qkv':<6}\tL{layer_idx:02d}\t{qkv}")
                        B, T, _ = output.shape
                        act = output.detach().float().view(B, T, n_heads, head_dim)
                        per_norms = act.norm(dim=-1).to(torch.float16)  # (B, T, n_heads) on GPU
                        if is_q:
                            self._q_act[self._s_idx, self._accum_idx, layer_idx].copy_(per_norms)
                        else:
                            self._kv_act[self._s_idx, self._accum_idx, layer_idx, kv_i].copy_(per_norms)
                    return fwd_hook

                def make_bwd_qkv(layer_idx, qkv, n_heads, head_dim):
                    is_q = (qkv == "q")
                    kv_i = self._KV_IDX.get(qkv, 0)
                    def bwd_hook(module, grad_input, grad_output):
                        if not module.training or not self._record_this_step:
                            return
                        # print(f"[bwd]\t{'qkv':<6}\tL{layer_idx:02d}\t{qkv}")
                        g = grad_output[0]
                        if g is None:
                            return
                        B, T, _ = g.shape
                        g_r = g.detach().float().view(B, T, n_heads, head_dim)
                        gn = g_r.norm(dim=-1).to(torch.float16)  # (B, T, n_heads)
                        if is_q:
                            self._q_grad[self._s_idx, self._accum_idx, layer_idx].copy_(gn)
                        else:
                            self._kv_grad[self._s_idx, self._accum_idx, layer_idx, kv_i].copy_(gn)
                    return bwd_hook

                self._hooks.append(proj.register_forward_hook(
                    make_fwd_qkv(i, qkv_name, n_heads, head_dim)))
                self._hooks.append(proj.register_full_backward_hook(
                    make_bwd_qkv(i, qkv_name, n_heads, head_dim)))

    # ---------------- flush (write npz) ----------------
    def _gather_slab(self, cpu_slice):
        """Gather a CPU fp16 tensor across DDP ranks as fp16 numpy.
        Returns np.ndarray of shape (R, *cpu_slice.shape) on rank 0; None elsewhere.
        For single-rank runs returns (1, *shape).

        Uses all_gather_object on numpy arrays to keep data on CPU — the slabs
        can be large and round-tripping through GPU for gather would balloon
        VRAM at flush time."""
        local_np = cpu_slice.contiguous().numpy()
        if self.world_size <= 1 or not is_ddp_initialized():
            return local_np[None, ...]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_np)
        if self.rank == 0:
            return np.stack(gathered, axis=0)
        return None

    def _reduce_sum(self, cpu_tensor):
        """Sum a CPU tensor across DDP ranks (running-sum moment accumulators).
        Returns a numpy array on rank 0; None on other ranks. Single-rank: copy.

        NOTE: returns a COPY, never a view of the accumulator storage — flush()
        zeros the accumulators after this call, which would otherwise wipe the
        returned data (these tensors alias their numpy() buffer)."""
        local_np = cpu_tensor.contiguous().numpy().copy()
        if self.world_size <= 1 or not is_ddp_initialized():
            return local_np
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_np)
        if self.rank == 0:
            return np.sum(np.stack(gathered, axis=0), axis=0)
        return None

    def flush(self, global_step=None, force=False):
        if not self._configured:
            return
        n = len(self._recorded_steps)
        # Decide whether to write now.
        if n == 0:
            return
        if not force and n < self._steps_per_file:
            return

        # Gather slabs across DDP ranks (every rank participates in collectives).
        hidden_act_g  = self._gather_slab(self._hidden_act[:n])     # (R, n, A, HS, L, B, T) or None
        hidden_grad_g = self._gather_slab(self._hidden_grad[:n])
        q_act_g   = self._gather_slab(self._q_act[:n])              # (R, n, A, L, B, T, H_q)
        q_grad_g  = self._gather_slab(self._q_grad[:n])
        kv_act_g  = self._gather_slab(self._kv_act[:n])             # (R, n, A, L, 2, B, T, H_kv)
        kv_grad_g = self._gather_slab(self._kv_grad[:n])

        # === exp3: gather attention weights (per-sample, stacked over ranks) and reduce moments. ===
        attnw_g = self._gather_slab(self._attnw_packed[:n])        # (R, n, A, L, B, H_q, P) uint8 or None
        # Moments are running SUMS over (accum, batch, pos): sum them across ranks (not stack).
        x_count_r = self._reduce_sum(self._x_count[:n])           # (n, L) or None
        x_sum_r   = self._reduce_sum(self._x_sum[:n])             # (n, L, C)
        x_xxT_r   = self._reduce_sum(self._x_xxT[:n])             # (n, L, C, C)
        g_count_r = self._reduce_sum(self._g_count[:n])           # (n, L, H_q)
        g_sum_r   = self._reduce_sum(self._g_sum[:n])             # (n, L, H_q, D)
        g_ggT_r   = self._reduce_sum(self._g_ggT[:n])            # (n, L, H_q, D, D)

        # Snapshot projection weights W_q/W_k/W_v per layer (same on all ranks; rank 0 reads them).
        # Weights are constant within this flush window's steps only if the model is fixed (exp3
        # eval is fixed). Stored once per window (the trained checkpoint is frozen during eval).
        W_q_np = W_k_np = W_v_np = None
        if self.rank == 0:
            W_q_list, W_k_list, W_v_list = [], [], []
            for spec in self._attn_specs:
                W_q_list.append(spec["proj"]["q"].weight.detach().float().cpu().numpy())
                W_k_list.append(spec["proj"]["k"].weight.detach().float().cpu().numpy())
                W_v_list.append(spec["proj"]["v"].weight.detach().float().cpu().numpy())
            W_q_np = np.stack(W_q_list, axis=0).astype(np.float16)   # (L, H_q*D, C)
            W_k_np = np.stack(W_k_list, axis=0).astype(np.float16)   # (L, H_kv*D, C)
            W_v_np = np.stack(W_v_list, axis=0).astype(np.float16)   # (L, H_kv*D, C)

        # Reset staging for the next window (every rank).
        recorded_steps = self._recorded_steps
        self._recorded_steps = []
        self._s_idx = 0
        # exp3: zero the moment accumulators for the next window.
        self._x_count.zero_(); self._x_sum.zero_(); self._x_xxT.zero_()
        self._g_count.zero_(); self._g_sum.zero_(); self._g_ggT.zero_()

        if self.rank != 0:
            return

        # Permute gathered axes into the on-disk layout expected by readers.
        #   hidden: (R, n=S, A, HS, L, B, T)   -> (S, A, R, B, HS, L, T)
        #   q:      (R, n=S, A, L, B, T, H_q)  -> (S, A, R, B, L, T, H_q)
        #   kv:     (R, n=S, A, L, 2, B, T, H_kv) -> (S, A, R, B, L, 2, T, H_kv)
        act = hidden_act_g.transpose(1, 2, 0, 5, 3, 4, 6).astype(np.float32)
        grd = hidden_grad_g.transpose(1, 2, 0, 5, 3, 4, 6).astype(np.float32)
        q_act  = q_act_g.transpose(1, 2, 0, 4, 3, 5, 6).astype(np.float32)
        q_grd  = q_grad_g.transpose(1, 2, 0, 4, 3, 5, 6).astype(np.float32)
        kv_act = kv_act_g.transpose(1, 2, 0, 5, 3, 4, 6, 7).astype(np.float32)
        kv_grd = kv_grad_g.transpose(1, 2, 0, 5, 3, 4, 6, 7).astype(np.float32)

        S = n
        A = self._A
        R = self.world_size
        B = self._B
        L = self._n_layer
        T = self._T
        HS = len(self._HIDDEN_STREAMS)
        H_q, H_kv = self._n_head, self._n_kv_head
        # hidden embed dim C — read from model config (was always a scalar in the output).
        n_elements = int(self.model.config.n_embd)

        layer_type_legend = self._layer_type_legend_arr
        layer_types_arr = self._layer_types_arr
        global_steps_arr = np.array(recorded_steps, dtype=np.int32)
        step_tag = f"step_{recorded_steps[0]}-{recorded_steps[-1]}"
        out_dir = os.path.join(self._logs_dir, "norms")

        # -------- HIDDEN (4 residual-stream points; axis after B is HS) --------
        def _quant_hidden(x):
            N = S * A * R * B * HS * L
            flat = x.reshape(N, T)
            q, sc, mn, oi, ov = _quantize_uint8_batched(flat, self._outlier_pct)
            K = oi.shape[1]
            return (q.reshape(S, A, R, B, HS, L, T),
                    sc.reshape(S, A, R, B, HS, L),
                    mn.reshape(S, A, R, B, HS, L),
                    oi.reshape(S, A, R, B, HS, L, K),
                    ov.reshape(S, A, R, B, HS, L, K))

        aq, asc, amn, aoi, aov = _quant_hidden(act)
        gq, gsc, gmn, goi, gov = _quant_hidden(grd)

        np.savez_compressed(
            os.path.join(out_dir, f"{step_tag}_hidden.npz"),
            act_q=aq, act_scale=asc, act_min=amn,
            act_outlier_idx=aoi, act_outlier_val=aov,
            grad_q=gq, grad_scale=gsc, grad_min=gmn,
            grad_outlier_idx=goi, grad_outlier_val=gov,
            global_steps=global_steps_arr,
            layer_types=layer_types_arr,
            layer_type_legend=layer_type_legend,
            stream_legend=np.array(self._HIDDEN_STREAMS),
            format_version=np.int32(2),
            n_elements=np.int32(n_elements),
            seq_len=np.int32(T),
            device_batch_size=np.int32(B),
            world_size=np.int32(R),
            grad_accum_steps=np.int32(A),
            outlier_pct=np.float32(self._outlier_pct),
        )

        # -------- ATTENTION --------
        if H_q > 0 and H_kv > 0:
            def _quant_attn_q(x):
                N = S * A * R * B * L
                flat = x.reshape(N, T * H_q)
                q, sc, mn, oi, ov = _quantize_uint8_batched(flat, self._outlier_pct)
                K = oi.shape[1]
                return (q.reshape(S, A, R, B, L, T, H_q),
                        sc.reshape(S, A, R, B, L),
                        mn.reshape(S, A, R, B, L),
                        oi.reshape(S, A, R, B, L, K),
                        ov.reshape(S, A, R, B, L, K))

            def _quant_attn_kv(x):
                N = S * A * R * B * L * 2
                flat = x.reshape(N, T * H_kv)
                q, sc, mn, oi, ov = _quantize_uint8_batched(flat, self._outlier_pct)
                K = oi.shape[1]
                return (q.reshape(S, A, R, B, L, 2, T, H_kv),
                        sc.reshape(S, A, R, B, L, 2),
                        mn.reshape(S, A, R, B, L, 2),
                        oi.reshape(S, A, R, B, L, 2, K),
                        ov.reshape(S, A, R, B, L, 2, K))

            qaq, qasc, qamn, qaoi, qaov = _quant_attn_q(q_act)
            qgq, qgsc, qgmn, qgoi, qgov = _quant_attn_q(q_grd)
            kvaq, kvasc, kvamn, kvaoi, kvaov = _quant_attn_kv(kv_act)
            kvgq, kvgsc, kvgmn, kvgoi, kvgov = _quant_attn_kv(kv_grd)

            np.savez_compressed(
                os.path.join(out_dir, f"{step_tag}_attn.npz"),
                q_act_q=qaq, q_act_scale=qasc, q_act_min=qamn,
                q_act_outlier_idx=qaoi, q_act_outlier_val=qaov,
                q_grad_q=qgq, q_grad_scale=qgsc, q_grad_min=qgmn,
                q_grad_outlier_idx=qgoi, q_grad_outlier_val=qgov,
                kv_act_q=kvaq, kv_act_scale=kvasc, kv_act_min=kvamn,
                kv_act_outlier_idx=kvaoi, kv_act_outlier_val=kvaov,
                kv_grad_q=kvgq, kv_grad_scale=kvgsc, kv_grad_min=kvgmn,
                kv_grad_outlier_idx=kvgoi, kv_grad_outlier_val=kvgov,
                global_steps=global_steps_arr,
                layer_types=layer_types_arr,
                layer_type_legend=layer_type_legend,
                kv_legend=np.array(["k", "v"]),
                n_head=np.int32(H_q),
                n_kv_head=np.int32(H_kv),
                head_dim=np.int32(self._head_dim),
                seq_len=np.int32(T),
                device_batch_size=np.int32(B),
                world_size=np.int32(R),
                grad_accum_steps=np.int32(A),
                outlier_pct=np.float32(self._outlier_pct),
            )

        # -------- exp3: ATTENTION WEIGHTS + THEORY-INPUT MOMENTS (_attnstat.npz) --------
        # Everything needed to evaluate the closed-form gradient theory on the realized
        # attention, alongside the experimental q/k/v grad norms in {step}_attn.npz.
        if H_q > 0 and attnw_g is not None:
            D = self._head_dim
            P = self._attn_P
            # attnw_g: (R, n=S, A, L, B, H_q, P) uint8 -> (S, A, R, B, L, H_q, P)
            attn_weights = attnw_g.transpose(1, 2, 0, 4, 3, 5, 6).copy()

            # Finalize input moments mu_x (S,L,C), Sigma_x (S,L,C,C) from running sums.
            cnt_x = np.maximum(x_count_r, 1.0)[..., None]                 # (S,L,1)
            mu_x = (x_sum_r / cnt_x).astype(np.float32)                   # (S,L,C)
            # Sigma_x = E[xx^T] - mu mu^T   (population covariance over tokens)
            ex_xxT = x_xxT_r / np.maximum(x_count_r, 1.0)[..., None, None]  # (S,L,C,C)
            Sigma_x = (ex_xxT - mu_x[..., :, None] * mu_x[..., None, :]).astype(np.float32)

            # Finalize grad moments mu_g (S,L,H,D), Sigma_g (S,L,H,D,D).
            cnt_g = np.maximum(g_count_r, 1.0)[..., None]                 # (S,L,H,1)
            mu_g = (g_sum_r / cnt_g).astype(np.float32)                   # (S,L,H,D)
            eg_ggT = g_ggT_r / np.maximum(g_count_r, 1.0)[..., None, None]  # (S,L,H,D,D)
            Sigma_g = (eg_ggT - mu_g[..., :, None] * mu_g[..., None, :]).astype(np.float32)

            # Projection weights are per-layer (model fixed during eval); broadcast a step axis
            # so the reader indexes them uniformly with everything else.
            W_q = np.broadcast_to(W_q_np[None], (S,) + W_q_np.shape).copy()  # (S,L,H_q*D,C)
            W_k = np.broadcast_to(W_k_np[None], (S,) + W_k_np.shape).copy()  # (S,L,H_kv*D,C)
            W_v = np.broadcast_to(W_v_np[None], (S,) + W_v_np.shape).copy()  # (S,L,H_kv*D,C)

            np.savez_compressed(
                os.path.join(out_dir, f"{step_tag}_attnstat.npz"),
                # --- attention weights A (causal, fp16, lower-triangular packed) ---
                attn_weights=attn_weights,            # (S, A, R, B, L, H_q, P) float16
                attn_dtype=np.array("float16"),       # dtype of attn_weights
                tri_packed=np.array(True),            # lower-triangular packed (see _tril_pack_f16)
                packed_len=np.int32(P),               # P = T*(T+1)//2
                # --- input moments of attn input x = norm(block_input) ---
                mu_x=mu_x, Sigma_x=Sigma_x,           # (S,L,C), (S,L,C,C)
                x_count=x_count_r.astype(np.int64),   # (S,L) tokens summed (for reference)
                # --- upstream per-head attention-output gradient moments g = dL/dy ---
                mu_g=mu_g, Sigma_g=Sigma_g,           # (S,L,H_q,D), (S,L,H_q,D,D)
                g_count=g_count_r.astype(np.int64),   # (S,L,H_q)
                # --- projection weights ---
                W_q=W_q, W_k=W_k, W_v=W_v,            # (S,L,*,C) fp16
                # --- metadata ---
                global_steps=global_steps_arr,
                layer_types=layer_types_arr,
                layer_type_legend=layer_type_legend,
                n_head=np.int32(H_q),
                n_kv_head=np.int32(H_kv),
                head_dim=np.int32(D),
                n_embd=np.int32(n_elements),
                seq_len=np.int32(T),
                device_batch_size=np.int32(B),
                world_size=np.int32(R),
                grad_accum_steps=np.int32(A),
                format_version=np.int32(1),
            )

    def remove_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
# === END NEW ===

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")

parser.add_argument("--no-smear",          action="store_true", help="disable smear (prev-token embedding mixing)")
parser.add_argument("--no-resid-lambdas",  action="store_true", help="disable per-layer resid_λ·x + x0_λ·x₀ mixing")
parser.add_argument("--no-value-residual", action="store_true", help="disable ResFormer value embeddings")
parser.add_argument("--no-backout",        action="store_true", help="disable mid-layer backout subtraction")
parser.add_argument("--no-rope",            action="store_true", help="disable RoPE (rotary positional embedding); QK-norm and x1.2 sharpening are kept")
parser.add_argument("--no-qknorm",          action="store_true", help="disable QK-norm on Q/K")
parser.add_argument("--no-qk-scale",        action="store_true", help="disable the x1.2 Q/K sharpening scale")

# === exp3 eval: which checkpoint to load and run the eval pass on ===
parser.add_argument("--step", type=int, default=None, help="checkpoint step to load (default: last step in the model-tag dir)")

# === NEW (norm monitoring): flags ===
parser.add_argument("--monitor-debug", action="store_true", help="print a debug line when the first monitor fwd hook fires")
parser.add_argument("--monitor-steps-per-file", type=int, default=5, help="flush one npz chunk per N recorded global steps")
parser.add_argument("--monitor-outlier-pct", type=float, default=0.01, help="fraction of largest values kept losslessly per sample")
parser.add_argument("--monitor-record-every-k-steps", type=int, default=20, help="only record a fwd/bwd pass every K global steps (K=1 records every step)")
# === END NEW ===

args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

# =============================================================================
# === NEW (norm monitoring): experiment_name + logs_dir, DDP-safe
# =============================================================================
# Rank 0 computes the timestamp; broadcast string so all ranks agree.
if master_process:
    _ts = datetime.now().strftime("%m%d%H%M")
    experiment_name = f"{args.run}_{_ts}"
else:
    experiment_name = None
if is_ddp_initialized():
    _obj = [experiment_name]
    dist.broadcast_object_list(_obj, src=0)
    experiment_name = _obj[0]
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(_project_root, "logs", experiment_name)
if master_process:
    os.makedirs(logs_dir, exist_ok=True)
print0(f"Experiment name: {experiment_name}")
print0(f"Logs dir: {logs_dir}")
# === END NEW ===

synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
        use_smear          = not args.no_smear,
        use_resid_lambdas  = not args.no_resid_lambdas,
        use_value_residual = not args.no_value_residual,
        use_backout        = not args.no_backout,
        use_rope           = not args.no_rope,
        use_qknorm         = not args.no_qknorm,
        use_qk_scale       = not args.no_qk_scale,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# === exp3 eval: ALWAYS load the trained model from a checkpoint and run a fixed eval pass. ===
# (We overwrite the freshly-initialized params with the checkpoint's state_dict. The model is
#  not trained here — see the commented-out optimizer.step() in the loop below.)
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
load_step = args.step if args.step is not None else find_last_step(checkpoint_dir)
print0(f"exp3 eval: loading checkpoint from {checkpoint_dir} at step {load_step}")
model_data, _optimizer_data_unused, meta_data = load_checkpoint(checkpoint_dir, load_step, device, load_optimizer=False, rank=ddp_rank)
# torch.compile checkpoints prepend "_orig_mod." to keys; strip it for a clean load.
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
model.load_state_dict(model_data, strict=True, assign=True)
del model_data # free up this memory after the copy
# `resuming` is kept for downstream code paths that reference it; eval never resumes optimizer state.
resuming = False

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# =============================================================================
# === exp3 eval: in-place swap attention forward to an EAGER softmax that
# === materializes the per-head attention weight matrix A (FA3/SDPA are fused
# === kernels and never expose A). We monkey-patch CausalSelfAttention.forward
# === at the class level so every layer uses it. The eager forward stashes
# === A on the module (`_last_attn_weights`) for the monitor's fwd hook to read.
# === It reuses the SAME parameters as the trained model — load_state_dict has
# === already populated c_q/c_k/c_v/c_proj — so no new model file is needed.
# =============================================================================
import torch.nn.functional as _F
from nanochat.gpt import CausalSelfAttention as _CSA, apply_rotary_emb as _apply_rotary_emb, norm as _qk_norm

def _eager_attention_forward(self, x, ve, cos_sin, window_size, kv_cache):
    """Eager re-implementation of CausalSelfAttention.forward that returns A.

    Mirrors nanochat/gpt.py CausalSelfAttention.forward exactly (projections,
    optional value-residual, optional RoPE / QK-norm / x1.2 scale), but computes
    attention with an explicit softmax so the weights A = softmax(scores) are
    materialized. Training path only (kv_cache is None); inference falls back to
    the original fused forward. When self._record_attn_weights is set, the realized
    per-head A (B, H_q, T, T) is detached and stashed on self._last_attn_weights.
    """
    if kv_cache is not None:
        # Inference path is unchanged — defer to the original fused implementation.
        return _ORIG_CSA_FORWARD(self, x, ve, cos_sin, window_size, kv_cache)

    B, T, C = x.size()
    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    if self.use_value_residual and ve is not None:
        ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * ve

    if self.use_rope:
        cos, sin = cos_sin
        q, k = _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)
    if self.use_qknorm:
        q, k = _qk_norm(q), _qk_norm(k)
    if self.use_qk_scale:
        q = q * 1.2
        k = k * 1.2

    # Move to (B, H, T, D) for batched matmul. Expand GQA kv-heads to query-head count
    # so A is per query head (B, H_q, T, T).
    n_rep = self.n_head // self.n_kv_head
    qh = q.transpose(1, 2)                                  # (B, H_q, T, D)
    kh = k.transpose(1, 2)                                  # (B, H_kv, T, D)
    vh = v.transpose(1, 2)                                  # (B, H_kv, T, D)
    if n_rep > 1:
        kh = kh.repeat_interleave(n_rep, dim=1)             # (B, H_q, T, D)
        vh = vh.repeat_interleave(n_rep, dim=1)

    scale = 1.0 / math.sqrt(self.head_dim)                  # same 1/sqrt(d) scale as FA3/SDPA
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale # (B, H_q, T, T); scores[b,h,s,r]=q_s·k_r/sqrt(d)

    # Causal mask (r > s masked) + optional sliding window (keep s-window <= r <= s).
    device = scores.device
    s_idx = torch.arange(T, device=device).unsqueeze(1)     # query position s (rows)
    r_idx = torch.arange(T, device=device).unsqueeze(0)     # key position r (cols)
    allowed = r_idx <= s_idx                                # causal
    left = window_size[0]
    if left is not None and left >= 0 and left < T:
        allowed = allowed & ((s_idx - r_idx) <= left)       # sliding window (left tokens)
    scores = scores.masked_fill(~allowed, float("-inf"))

    A = _F.softmax(scores, dim=-1)                          # (B, H_q, T, T); softmax over keys r
    if getattr(self, "_record_attn_weights", False):
        self._last_attn_weights = A.detach()                # for the monitor's attn fwd hook
    yh = torch.matmul(A, vh)                                # (B, H_q, T, D)
    y = yh.transpose(1, 2).contiguous().view(B, T, -1)      # (B, T, H_q*D)
    y = self.c_proj(y)
    return y

_ORIG_CSA_FORWARD = _CSA.forward
_CSA.forward = _eager_attention_forward
# give every attention module the runtime attributes the eager path / monitor use
# (`model` here is still the uncompiled model; orig_model is bound to it just below)
for _blk in model.transformer.h:
    if getattr(_blk, "attn", None) is not None:
        _blk.attn._record_attn_weights = False
        _blk.attn._last_attn_weights = None
print0("exp3 eval: patched CausalSelfAttention.forward -> eager (materializes attention weights A)")

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)

# =============================================================================
# === NEW (norm monitoring): attach hooks on the uncompiled model
# =============================================================================
monitor = GradientBiasMonitor(
    orig_model, logs_dir,
    rank=ddp_rank, world_size=ddp_world_size,
    steps_per_file=args.monitor_steps_per_file,
    outlier_pct=args.monitor_outlier_pct,
    record_every_k_steps=args.monitor_record_every_k_steps,
    debug=args.monitor_debug,
)
print0(f"GradientBiasMonitor attached: {len(orig_model.transformer.h)} layers "
       f"(layer_types={monitor._layer_types})")
# exp3: estimate attention-weight storage so the user can size --monitor-steps-per-file.
_T = args.max_seq_len
_P = _T * (_T + 1) // 2
_attn_bytes_per_sample = _P * monitor._n_head * monitor._n_layer * 2  # fp16 = 2 bytes/value, all layers, all q-heads
print0(f"exp3 attn-weight storage estimate: packed fp16 A ~= {_attn_bytes_per_sample/1e6:.2f} MB / sample "
       f"(P={_P} x H_q={monitor._n_head} x L={monitor._n_layer}); "
       f"x grad_accum x world x device_batch x monitor_steps_per_file per file.")
# === END NEW ===

# exp3 eval: do NOT torch.compile — the eager attention monkey-patch + fwd/bwd hooks that
# capture A and the moments must run in eager mode to be reliable. (model stays uncompiled.)
# model = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# === NEW (norm monitoring): now that A/B/T are all known, allocate the monitor's staging slabs. ===
monitor.configure(grad_accum_steps=grad_accum_steps,
                  device_batch_size=args.device_batch_size,
                  seq_len=args.max_seq_len)
# === END NEW ===

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # === exp3 eval: NO checkpoint saving — this is an eval pass on a fixed model, and we must
    # not overwrite the trained checkpoint we loaded from. (Original save block left below,
    # commented out, for an easy diff vs exp2_base_train.py.)
    # if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
    #     save_checkpoint(
    #         checkpoint_dir, step,
    #         orig_model.state_dict(), optimizer.state_dict(),
    #         { "step": step, "val_bpb": val_bpb, "model_config": model_config_kwargs,
    #           "user_config": user_config, "device_batch_size": args.device_batch_size,
    #           "max_seq_len": args.max_seq_len, "total_batch_size": total_batch_size,
    #           "dataloader_state_dict": dataloader_state_dict,
    #           "loop_state": {"min_val_bpb": min_val_bpb, "smooth_train_loss": smooth_train_loss,
    #                          "total_training_time": total_training_time}, },
    #         rank=ddp_rank,
    #     )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    # === NEW (norm monitoring): reset per-step accum index / fwd cache ===
    monitor.set_step(step)
    # === END NEW ===
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        # === NEW (norm monitoring): advance accum index so next micro-step caches separately ===
        monitor.advance_accum()
        # === END NEW ===
    # === exp3 eval: the model stays FIXED — NO optimizer step. ===
    # We keep loss.backward() above (so gradients are produced for the monitor/hooks) and
    # model.zero_grad() below (so grads are cleared each step), but the optimizer update and
    # LR/momentum/weight-decay scheduling are commented out. Left in place for an easy diff
    # vs exp2_base_train.py.
    lrm = get_lr_multiplier(step)  # kept for logging only
    # muon_momentum = get_muon_momentum(step)
    # muon_weight_decay = get_weight_decay(step)
    # for group in optimizer.param_groups:
    #     group["lr"] = group["initial_lr"] * lrm
    #     if group['kind'] == 'muon':
    #         group["momentum"] = muon_momentum
    #         group["weight_decay"] = muon_weight_decay
    # if scaler is not None:
    #     scaler.unscale_(optimizer)
    #     # In distributed training, all ranks must agree on whether to skip the step.
    #     # Each rank may independently encounter inf/nan gradients, so we all-reduce
    #     # the found_inf flag (MAX = if any rank found inf, all ranks skip).
    #     if is_ddp_initialized():
    #         for v in scaler._found_inf_per_device(optimizer).values():
    #             dist.all_reduce(v, op=dist.ReduceOp.MAX)
    #     scaler.step(optimizer)
    #     scaler.update()
    # else:
    #     optimizer.step()
    model.zero_grad(set_to_none=True)
    # === NEW (norm monitoring): flush collected records to parquet ===
    monitor.flush(step)
    # === END NEW ===
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# === NEW (norm monitoring): force-drain remaining records, then remove hooks ===
monitor.flush(step, force=True)
monitor.remove_hooks()
# === END NEW ===

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
