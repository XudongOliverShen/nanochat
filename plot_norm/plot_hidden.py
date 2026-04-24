import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
sys.path.insert(0, os.path.abspath('.'))

import itertools
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc

from read_norms import read_hidden

@dataclass
class View:
    """Flat (step, accum, traj, layer, pos) indexing over a single norm array.

    arr           : (S, A, Traj, L, T) float32
    global_steps  : (S,) int     — map global_step -> axis-0 index
    rel_positions : (T,) int     — x-axis values; `-1` is last token
    layers        : list[int]    — 0..L-1 (kept for convenience)
    trajs         : list[int]    — 0..Traj-1 = rank*B + sample
    """
    arr: np.ndarray
    global_steps: np.ndarray
    rel_positions: np.ndarray
    layers: List[int]
    trajs: List[int]

    def _s(self, global_step):
        hits = np.where(self.global_steps == int(global_step))[0]
        if hits.size == 0:
            raise KeyError(f'global_step {global_step} not in loaded range '
                           f'{self.global_steps.tolist()}')
        return int(hits[0])

    def row(self, global_step, accum_idx, traj_idx, layer):
        s = self._s(global_step)
        return self.arr[s, int(accum_idx), int(traj_idx), int(layer), :].astype(np.float32)


def _smooth(norms, window_avg):
    if window_avg <= 1:
        return norms
    return (pd.Series(norms)
              .rolling(window_avg, center=True, min_periods=1)
              .mean()
              .to_numpy(dtype=np.float32))


def _padded_range(norms, pad_frac=0.05):
    lo, hi = float(norms.min()), float(norms.max())
    pad = (hi - lo) * pad_frac if hi > lo else abs(lo) * pad_frac or 0.01
    return lo - pad, hi + pad


def _global_padded(norms_list, pad_frac=0.05):
    lo = min(float(n.min()) for n in norms_list)
    hi = max(float(n.max()) for n in norms_list)
    pad = (hi - lo) * pad_frac if hi > lo else abs(lo) * pad_frac or 0.01
    return lo - pad, hi + pad


def _aligned_range(v, lo, hi, frac):
    span = max((v - lo) / frac, (hi - v) / (1 - frac)) * 1.05
    return v - frac * span, v + (1 - frac) * span


def _tickvals(pos_min):
    step = max(1, 2 ** int(round(np.log2(abs(pos_min) / 8))))
    start = (pos_min // step) * step
    return [v for v in range(start, 0, step) if v != -1] + [-1]

def plot_single(view: View, global_step, accum_idx, traj_idx, layer,
                title: str, window_avg: int = 1):
    """Plot norms for a single (global_step, accum_idx, traj_idx, layer) trajectory.

    window_avg : centered rolling-average window applied before plotting (1 = no smoothing).
    """
    norms = _smooth(view.row(global_step, accum_idx, traj_idx, layer), window_avg)
    pos = view.rel_positions

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pos, y=norms, mode='lines', line=dict(width=1.5), showlegend=False,
        hovertemplate='pos: %{x}<br>norm: %{y:.4f}<extra></extra>',
    ))
    fig.add_vline(x=-1, line=dict(color='red', width=1, dash='dash'),
                  annotation_text='last token (−1)', annotation_position='top left',
                  annotation=dict(font=dict(color='red', size=11)))
    smooth_tag = f', window_avg={window_avg}' if window_avg > 1 else ''
    fig.update_layout(
        title=f'{title}  —  layer {layer}, step {global_step}, '
              f'accum {accum_idx}, traj {traj_idx}{smooth_tag}',
        xaxis=dict(title='Relative token position  (−1 = last token)',
                   tickmode='array', tickvals=_tickvals(int(pos.min()))),
        yaxis=dict(title='Norm'),
        height=400, margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def plot_single_act_grad_norm(v_act: Optional[View], v_grad: Optional[View],
                              global_step, accum_idx, traj_idx, layer,
                              title: str, window_avg: int = 1):
    """Overlay act (left y-axis) and grad (right y-axis) for one trajectory.
    Either view may be None; axes are aligned so the pos=−1 values meet."""
    act_norms = grad_norms = None
    if v_act is not None:
        act_norms = _smooth(v_act.row(global_step, accum_idx, traj_idx, layer), window_avg)
    if v_grad is not None:
        grad_norms = _smooth(v_grad.row(global_step, accum_idx, traj_idx, layer), window_avg)
    both = act_norms is not None and grad_norms is not None

    if both:
        ymin_a, ymax_a = _padded_range(act_norms)
        ymin_g, ymax_g = _padded_range(grad_norms)
        va, vg = float(act_norms[-1]), float(grad_norms[-1])
        fa = (va - ymin_a) / (ymax_a - ymin_a)
        fg = (vg - ymin_g) / (ymax_g - ymin_g)
        tgt = max(0.05, min(0.95, (fa + fg) / 2))
        ymin_a, ymax_a = _aligned_range(va, ymin_a, ymax_a, tgt)
        ymin_g, ymax_g = _aligned_range(vg, ymin_g, ymax_g, tgt)
    elif act_norms is not None:
        ymin_a, ymax_a = _padded_range(act_norms)
    else:
        ymin_g, ymax_g = _padded_range(grad_norms)

    pos = (v_act or v_grad).rel_positions
    fig = go.Figure()
    if act_norms is not None:
        fig.add_trace(go.Scatter(x=pos, y=act_norms, mode='lines', name='act norm',
                                 yaxis='y1', line=dict(color='#1f77b4', width=1.0, dash='dash'),
                                 hovertemplate='pos: %{x}<br>act norm: %{y:.4f}<extra></extra>'))
    if grad_norms is not None:
        fig.add_trace(go.Scatter(x=pos, y=grad_norms, mode='lines', name='grad norm',
                                 yaxis='y2' if both else 'y1',
                                 line=dict(color='#1f77b4', width=2.5),
                                 hovertemplate='pos: %{x}<br>grad norm: %{y:.4f}<extra></extra>'))

    smooth_tag = f', window_avg={window_avg}' if window_avg > 1 else ''
    if both or act_norms is not None:
        y1_title, y1_range = 'Act norm', [ymin_a, ymax_a]
    else:
        y1_title, y1_range = 'Grad norm', [ymin_g, ymax_g]

    layout = dict(
        title=f'{title}  —  layer {layer}, step {global_step}, '
              f'accum {accum_idx}, traj {traj_idx}{smooth_tag}',
        xaxis=dict(title='Relative token position  (−1 = last token)',
                   tickmode='array', tickvals=_tickvals(int(pos.min()))),
        yaxis=dict(title=y1_title, range=y1_range),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
        height=450, margin=dict(l=60, r=80, t=60, b=60),
    )
    if both:
        layout['yaxis2'] = dict(title='Grad norm', range=[ymin_g, ymax_g],
                                overlaying='y', side='right')
    fig.update_layout(**layout)
    return fig


def plot_single_act_grad_norm_normalized(v_act: Optional[View], v_grad: Optional[View],
                                         global_step, accum_idx, traj_idx, layer,
                                         title: str, window_avg: int = 1):
    """Same as plot_single_act_grad_norm but each curve is divided by its
    pos=−1 value before plotting, so both curves meet at (−1, 1.0)."""
    def _get(view):
        if view is None:
            return None
        norms = _smooth(view.row(global_step, accum_idx, traj_idx, layer), window_avg)
        ref = float(norms[-1])
        if ref == 0:
            raise ValueError('norm at pos=−1 is zero; cannot normalize')
        return norms / ref

    act_norms = _get(v_act)
    grad_norms = _get(v_grad)
    both = act_norms is not None and grad_norms is not None

    if both:
        ymin_a, ymax_a = _padded_range(act_norms)
        ymin_g, ymax_g = _padded_range(grad_norms)
        va, vg = float(act_norms[-1]), float(grad_norms[-1])
        fa = (va - ymin_a) / (ymax_a - ymin_a)
        fg = (vg - ymin_g) / (ymax_g - ymin_g)
        tgt = max(0.05, min(0.95, (fa + fg) / 2))
        ymin_a, ymax_a = _aligned_range(va, ymin_a, ymax_a, tgt)
        ymin_g, ymax_g = _aligned_range(vg, ymin_g, ymax_g, tgt)
    elif act_norms is not None:
        ymin_a, ymax_a = _padded_range(act_norms)
    else:
        ymin_g, ymax_g = _padded_range(grad_norms)

    pos = (v_act or v_grad).rel_positions
    fig = go.Figure()
    if act_norms is not None:
        fig.add_trace(go.Scatter(x=pos, y=act_norms, mode='lines', name='act norm (norm.)',
                                 yaxis='y1', line=dict(color='#1f77b4', width=1.0, dash='dash'),
                                 hovertemplate='pos: %{x}<br>act (norm.): %{y:.4f}<extra></extra>'))
    if grad_norms is not None:
        fig.add_trace(go.Scatter(x=pos, y=grad_norms, mode='lines', name='grad norm (norm.)',
                                 yaxis='y2' if both else 'y1',
                                 line=dict(color='#1f77b4', width=2.5),
                                 hovertemplate='pos: %{x}<br>grad (norm.): %{y:.4f}<extra></extra>'))

    smooth_tag = f', window_avg={window_avg}' if window_avg > 1 else ''
    if both or act_norms is not None:
        y1_title, y1_range = 'Act norm / act norm at pos=−1', [ymin_a, ymax_a]
    else:
        y1_title, y1_range = 'Grad norm / grad norm at pos=−1', [ymin_g, ymax_g]

    layout = dict(
        title=f'{title}  —  layer {layer}, step {global_step}, '
              f'accum {accum_idx}, traj {traj_idx}{smooth_tag}',
        xaxis=dict(title='Relative token position  (−1 = last token)',
                   tickmode='array', tickvals=_tickvals(int(pos.min()))),
        yaxis=dict(title=y1_title, range=y1_range),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
        height=450, margin=dict(l=60, r=80, t=60, b=60),
    )
    if both:
        layout['yaxis2'] = dict(title='Grad norm / grad norm at pos=−1',
                                range=[ymin_g, ymax_g], overlaying='y', side='right')
    fig.update_layout(**layout)
    return fig

def plot_all_steps_act_grad_norm_normalized(v_act: Optional[View], v_grad: Optional[View],
                                            global_step, accum_idx, traj_idx, layer: int,
                                            title: str, window_avg: int = 1):
    """Overlay one normalized line per combo of (global_step, accum_idx, traj_idx).

    Each parameter may be:
      - scalar  → fixed
      - list    → iterate
      - None    → use all available values
    """
    v_ref = v_act if v_act is not None else v_grad

    def _to_list(val, universe):
        if val is None:
            return [int(x) for x in universe]
        return [int(x) for x in (val if isinstance(val, list) else [val])]

    steps  = _to_list(global_step, v_ref.global_steps.tolist())
    accums = _to_list(accum_idx,   list(range(v_ref.arr.shape[1])))
    trajs  = _to_list(traj_idx,    v_ref.trajs)
    combos = list(itertools.product(steps, accums, trajs))

    if len(combos) == 1:
        s, a, t = combos[0]
        return plot_single_act_grad_norm_normalized(v_act, v_grad, s, a, t, layer, title, window_avg)

    def _get(view, s, a, t):
        if view is None:
            return None
        try:
            norms = _smooth(view.row(s, a, t, layer), window_avg)
        except KeyError:
            return None
        ref = float(norms[-1])
        return None if ref == 0 else norms / ref

    records = []
    for s, a, t in combos:
        act_n = _get(v_act, s, a, t)
        grad_n = _get(v_grad, s, a, t)
        act_ok  = v_act is None or act_n is not None
        grad_ok = v_grad is None or grad_n is not None
        if not (act_ok and grad_ok):
            continue
        records.append((s, a, t, act_n, grad_n))
    if not records:
        raise ValueError('No valid data found for any combination.')

    both = v_act is not None and v_grad is not None
    if both:
        ymin_a, ymax_a = _global_padded([r[3] for r in records])
        ymin_g, ymax_g = _global_padded([r[4] for r in records])
        fa = (1.0 - ymin_a) / (ymax_a - ymin_a)
        fg = (1.0 - ymin_g) / (ymax_g - ymin_g)
        tgt = max(0.05, min(0.95, (fa + fg) / 2))
        ymin_a, ymax_a = _aligned_range(1.0, ymin_a, ymax_a, tgt)
        ymin_g, ymax_g = _aligned_range(1.0, ymin_g, ymax_g, tgt)
    elif v_act is not None:
        ymin_1, ymax_1 = _global_padded([r[3] for r in records])
    else:
        ymin_1, ymax_1 = _global_padded([r[4] for r in records])

    n = len(records)
    colors = pc.sample_colorscale('Plasma', [i / max(n - 1, 1) for i in range(n)])
    vary_step  = len({r[0] for r in records}) > 1
    vary_accum = len({r[1] for r in records}) > 1
    vary_traj  = len({r[2] for r in records}) > 1

    def _lbl(s, a, t):
        parts = []
        if vary_step:  parts.append(f'step={s}')
        if vary_accum: parts.append(f'accum={a}')
        if vary_traj:  parts.append(f'traj={t}')
        return ', '.join(parts) or f'step={s},accum={a},traj={t}'

    pos = v_ref.rel_positions
    fig = go.Figure()
    for i, (s, a, t, act_n, grad_n) in enumerate(records):
        color = colors[i]
        lbl = _lbl(s, a, t)
        if act_n is not None:
            fig.add_trace(go.Scatter(x=pos, y=act_n, mode='lines', name=f'act  {lbl}',
                                     yaxis='y1', legendgroup=lbl,
                                     line=dict(color=color, width=1.0, dash='dash'),
                                     hovertemplate=f'pos: %{{x}}<br>act (norm.) [{lbl}]: %{{y:.4f}}<extra></extra>'))
        if grad_n is not None:
            fig.add_trace(go.Scatter(x=pos, y=grad_n, mode='lines', name=f'grad {lbl}',
                                     yaxis='y2' if both else 'y1', legendgroup=lbl,
                                     line=dict(color=color, width=2.0),
                                     hovertemplate=f'pos: %{{x}}<br>grad (norm.) [{lbl}]: %{{y:.4f}}<extra></extra>'))

    smooth_tag = f', window_avg={window_avg}' if window_avg > 1 else ''
    if both:
        y1_title, y1_range = 'Act norm / act norm at pos=−1', [ymin_a, ymax_a]
    elif v_act is not None:
        y1_title, y1_range = 'Act norm / act norm at pos=−1', [ymin_1, ymax_1]
    else:
        y1_title, y1_range = 'Grad norm / grad norm at pos=−1', [ymin_1, ymax_1]

    layout = dict(
        title=f'{title}  —  layer {layer}{smooth_tag}',
        xaxis=dict(title='Relative token position  (−1 = last token)',
                   tickmode='array', tickvals=_tickvals(int(pos.min()))),
        yaxis=dict(title=y1_title, range=y1_range),
        legend=dict(x=1.08, y=1.0, xanchor='left', yanchor='top'),
        height=500, margin=dict(l=60, r=200, t=60, b=60),
    )
    if both:
        layout['yaxis2'] = dict(title='Grad norm / grad norm at pos=−1',
                                range=[ymin_g, ymax_g], overlaying='y', side='right')
    fig.update_layout(**layout)
    return fig


def plot_all_layers_act_grad_norm_normalized(v_act: Optional[View], v_grad: Optional[View],
                                             global_step, accum_idx, traj_idx, layer,
                                             title: str, window_avg: int = 1):
    """Overlay one normalized line per layer at a fixed (step, accum, traj)."""
    layers_list = layer if isinstance(layer, list) else [layer]
    if len(layers_list) == 1:
        return plot_single_act_grad_norm_normalized(
            v_act, v_grad, global_step, accum_idx, traj_idx, layers_list[0], title, window_avg)

    def _get(view, lyr):
        if view is None:
            return None
        norms = _smooth(view.row(global_step, accum_idx, traj_idx, lyr), window_avg)
        ref = float(norms[-1])
        return None if ref == 0 else norms / ref

    records = []
    for lyr in layers_list:
        act_n = _get(v_act, lyr)
        grad_n = _get(v_grad, lyr)
        act_ok  = v_act is None or act_n is not None
        grad_ok = v_grad is None or grad_n is not None
        if not (act_ok and grad_ok):
            continue
        records.append((lyr, act_n, grad_n))
    if not records:
        raise ValueError('No valid data found for any layer.')

    both = v_act is not None and v_grad is not None
    if both:
        ymin_a, ymax_a = _global_padded([r[1] for r in records])
        ymin_g, ymax_g = _global_padded([r[2] for r in records])
        fa = (1.0 - ymin_a) / (ymax_a - ymin_a)
        fg = (1.0 - ymin_g) / (ymax_g - ymin_g)
        tgt = max(0.05, min(0.95, (fa + fg) / 2))
        ymin_a, ymax_a = _aligned_range(1.0, ymin_a, ymax_a, tgt)
        ymin_g, ymax_g = _aligned_range(1.0, ymin_g, ymax_g, tgt)
    elif v_act is not None:
        ymin_1, ymax_1 = _global_padded([r[1] for r in records])
    else:
        ymin_1, ymax_1 = _global_padded([r[2] for r in records])

    n = len(records)
    colors = pc.sample_colorscale('Plasma', [i / max(n - 1, 1) for i in range(n)])
    pos = (v_act or v_grad).rel_positions

    fig = go.Figure()
    for i, (lyr, act_n, grad_n) in enumerate(records):
        color = colors[i]
        lbl = f'layer {lyr}'
        if act_n is not None:
            fig.add_trace(go.Scatter(x=pos, y=act_n, mode='lines', name=f'act  {lbl}',
                                     yaxis='y1', legendgroup=lbl,
                                     line=dict(color=color, width=1.0, dash='dash'),
                                     hovertemplate=f'pos: %{{x}}<br>act (norm.) [{lbl}]: %{{y:.4f}}<extra></extra>'))
        if grad_n is not None:
            fig.add_trace(go.Scatter(x=pos, y=grad_n, mode='lines', name=f'grad {lbl}',
                                     yaxis='y2' if both else 'y1', legendgroup=lbl,
                                     line=dict(color=color, width=2.0),
                                     hovertemplate=f'pos: %{{x}}<br>grad (norm.) [{lbl}]: %{{y:.4f}}<extra></extra>'))

    smooth_tag = f', window_avg={window_avg}' if window_avg > 1 else ''
    if both:
        y1_title, y1_range = 'Act norm / act norm at pos=−1', [ymin_a, ymax_a]
    elif v_act is not None:
        y1_title, y1_range = 'Act norm / act norm at pos=−1', [ymin_1, ymax_1]
    else:
        y1_title, y1_range = 'Grad norm / grad norm at pos=−1', [ymin_1, ymax_1]

    layout = dict(
        title=f'{title}  —  step {global_step}, accum {accum_idx}, traj {traj_idx}{smooth_tag}',
        xaxis=dict(title='Relative token position  (−1 = last token)',
                   tickmode='array', tickvals=_tickvals(int(pos.min()))),
        yaxis=dict(title=y1_title, range=y1_range),
        legend=dict(x=1.08, y=1.0, xanchor='left', yanchor='top'),
        height=500, margin=dict(l=60, r=200, t=60, b=60),
    )
    if both:
        layout['yaxis2'] = dict(title='Grad norm / grad norm at pos=−1',
                                range=[ymin_g, ymax_g], overlaying='y', side='right')
    fig.update_layout(**layout)
    return fig

NORMS_DIR  = '/home/svu/xudong_shen/myscratch/nanochat/logs/speedrun_04241621/norms'

STEP_START = 0    # first global_step to include (inclusive)
STEP_END   = 3    # last  global_step to include (inclusive)

h = read_hidden(NORMS_DIR, STEP_START, STEP_END)

S, A, R, B, L, T = h.act.shape
# Flatten (R, B) -> Traj to mirror the reference notebook's single traj_idx.
act_flat  = h.act.reshape(S, A, R * B, L, T)
grad_flat = h.grad.reshape(S, A, R * B, L, T)

# Last token of the sequence lives at pos = -1.
rel_positions = np.arange(T, dtype=np.int64) - T
layers = list(range(L))
trajs  = list(range(R * B))

v_act  = View(act_flat,  h.global_steps, rel_positions, layers, trajs)
v_grad = View(grad_flat, h.global_steps, rel_positions, layers, trajs)

print(f'Steps:  {h.global_steps[0]} → {h.global_steps[-1]}  ({S} total)')
print(f'Layers: 0 → {L-1}  ({L} total, types={h.layer_types})')
print(f'Grad-accum micro-steps: {A}')
print(f'Trajectories (rank × sample): {R} × {B} = {R*B}')
print(f'Sequence length T: {T}    embed dim C: {h.n_elements}')
print(f'act array bytes:  {act_flat.nbytes/1e6:.1f} MB   grad array bytes: {grad_flat.nbytes/1e6:.1f} MB')

plot_single(v_act, global_step=3, accum_idx=0, traj_idx=0, layer=0,
            title='Hidden activation norm').show()

plot_single(v_grad, global_step=3, accum_idx=0, traj_idx=0, layer=0,
            title='Hidden gradient norm').show()

plot_single_act_grad_norm(
    v_act, v_grad,
    global_step=0, accum_idx=0, traj_idx=0, layer=0,
    title='Act & Grad norm', window_avg=200,
).show()

plot_single_act_grad_norm_normalized(
    v_act, v_grad,
    global_step=0, accum_idx=0, traj_idx=0, layer=0,
    title='Act & Grad norm (normalized to pos=−1)', window_avg=200,
).show()

# One act curve per step, for a couple of trajectories
plot_all_steps_act_grad_norm_normalized(
    v_act, None,
    global_step=None, accum_idx=0, traj_idx=[0, 1], layer=0,
    title='Act norm (norm., all steps)', window_avg=200,
).show()

# Same but for grad
plot_all_steps_act_grad_norm_normalized(
    None, v_grad,
    global_step=None, accum_idx=0, traj_idx=[0, 1], layer=0,
    title='Grad norm (norm., all steps)', window_avg=200,
).show()

plot_all_layers_act_grad_norm_normalized(
    v_act, None,
    global_step=int(h.global_steps[-1]), accum_idx=0, traj_idx=0, layer=layers,
    title='Act norm (norm., all layers)', window_avg=200,
).show()

plot_all_layers_act_grad_norm_normalized(
    None, v_grad,
    global_step=int(h.global_steps[-1]), accum_idx=0, traj_idx=0, layer=layers,
    title='Grad norm (norm., all layers)', window_avg=200,
).show()
