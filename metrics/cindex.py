# src/metrics/cindex.py
import torch

def harrell_cindex(durations: torch.Tensor, events: torch.Tensor, scores: torch.Tensor) -> float:
    t = durations.detach().cpu().numpy()
    e = events.detach().cpu().numpy().astype(bool)
    s = scores.detach().cpu().numpy()
    n = len(t)
    concordant = 0.0
    permissible = 0.0
    ties = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if e[i] and t[i] < t[j]:
                permissible += 1
                if s[i] == s[j]:
                    ties += 1
                elif s[i] > s[j]:
                    concordant += 1
            elif e[j] and t[j] < t[i]:
                permissible += 1
                if s[j] == s[i]:
                    ties += 1
                elif s[j] > s[i]:
                    concordant += 1
    if permissible == 0:
        return float('nan')
    return float((concordant + 0.5 * ties) / permissible)
