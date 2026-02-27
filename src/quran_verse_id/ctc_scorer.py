"""
CTC forced-alignment scoring (log-domain forward algorithm).

Given:
- log_probs: [T, V] (log-softmax over vocab for each frame)
- target_ids: [L] token IDs (no blanks)

We compute log P(target | log_probs) under standard CTC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import math
import torch


@dataclass
class CTCScoreResult:
    logp: float
    per_frame_len: int
    target_len: int


def _logsumexp3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Numerically stable logsumexp for 3 scalars (tensor)."""
    m = torch.maximum(a, torch.maximum(b, c))
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m) + torch.exp(c - m))


def ctc_forward_logp(
    log_probs: torch.Tensor,
    target_ids: Sequence[int],
    *,
    blank_id: int,
) -> CTCScoreResult:
    """
    Compute CTC forward log-prob for a single target.

    Args:
        log_probs: [T, V] log-softmax probabilities.
        target_ids: [L] token ids.
        blank_id: blank token id.

    Returns:
        CTCScoreResult with total log probability.
    """
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must be [T,V], got {tuple(log_probs.shape)}")
    T, V = log_probs.shape
    if T == 0:
        raise ValueError("log_probs has T=0")
    if not target_ids:
        # Probability of emitting nothing under CTC is the path of all blanks.
        logp = float(log_probs[:, blank_id].sum().item())
        return CTCScoreResult(logp=logp, per_frame_len=T, target_len=0)

    device = log_probs.device
    dtype = log_probs.dtype

    # Build extended sequence with blanks: [blank, t1, blank, t2, ..., tn, blank]
    ext: List[int] = [blank_id]
    for tid in target_ids:
        ext.append(int(tid))
        ext.append(blank_id)
    S = len(ext)  # 2L + 1

    # alpha: [S], initialize to -inf
    neg_inf = torch.tensor(-1e9, device=device, dtype=dtype)
    alpha = neg_inf.repeat(S)

    # t=0 init
    alpha[0] = log_probs[0, blank_id]
    alpha[1] = log_probs[0, ext[1]]  # first token
    # others stay -inf

    for t in range(1, T):
        prev = alpha
        alpha = neg_inf.repeat(S)

        for s in range(S):
            stay = prev[s]
            step1 = prev[s - 1] if s - 1 >= 0 else neg_inf
            step2 = neg_inf
            if s - 2 >= 0:
                # Can skip over a blank, unless it would merge repeats
                if ext[s] != blank_id and ext[s] != ext[s - 2]:
                    step2 = prev[s - 2]

            alpha[s] = log_probs[t, ext[s]] + _logsumexp3(stay, step1, step2)

    # End can be in last blank or last label
    logp = torch.logaddexp(alpha[S - 1], alpha[S - 2]).item()
    return CTCScoreResult(logp=float(logp), per_frame_len=T, target_len=len(target_ids))
