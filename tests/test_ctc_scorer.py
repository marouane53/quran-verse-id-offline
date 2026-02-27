import itertools
import math
import torch

from quran_verse_id.ctc_scorer import ctc_forward_logp


def brute_ctc_logp(log_probs, target_ids, blank_id):
    """
    Brute force enumerate all alignments for tiny T and tiny vocab.
    Only used in tests.
    """
    T, V = log_probs.shape
    target = list(target_ids)

    # Generate all paths of length T over vocab [0..V-1]
    total = None
    for path in itertools.product(range(V), repeat=T):
        # Collapse repeats, remove blanks
        collapsed = []
        prev = None
        for p in path:
            if p == blank_id:
                prev = p
                continue
            if p != prev:
                collapsed.append(p)
            prev = p
        if collapsed == target:
            lp = 0.0
            for t, p in enumerate(path):
                lp += float(log_probs[t, p].item())
            total = lp if total is None else float(torch.logaddexp(torch.tensor(total), torch.tensor(lp)).item())
    return total if total is not None else -1e9


def test_ctc_forward_matches_bruteforce():
    torch.manual_seed(0)
    T = 3
    V = 3
    blank = 0
    logits = torch.randn(T, V)
    log_probs = torch.log_softmax(logits, dim=-1)

    target = [1, 2]
    res = ctc_forward_logp(log_probs, target, blank_id=blank)
    brute = brute_ctc_logp(log_probs, target, blank)
    assert abs(res.logp - brute) < 1e-4
