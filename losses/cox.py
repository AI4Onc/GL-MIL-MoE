from __future__ import annotations
import torch

def _efron_partial_ll(log_risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(durations, descending=True)
    s = log_risk[order]
    t = durations[order]
    e = events[order].float()

    exp_s = torch.exp(s)
    cum_exp = torch.cumsum(exp_s, dim=0)

    event_idx = torch.nonzero(e > 0, as_tuple=False).view(-1)
    if event_idx.numel() == 0:
        return torch.sum(log_risk) * 0.0
    evt_times = t[event_idx]
    change = torch.ones_like(evt_times, dtype=torch.bool)
    change[1:] = evt_times[1:] != evt_times[:-1]
    group_starts = torch.nonzero(change, as_tuple=False).view(-1)
    group_starts = torch.cat([group_starts, torch.tensor([len(event_idx)], device=group_starts.device)])

    pll = torch.zeros((), device=log_risk.device, dtype=log_risk.dtype)
    start_ptr = 0
    for g in range(group_starts.numel() - 1):
        g_start = group_starts[g].item()
        g_end = group_starts[g + 1].item()
        idx_g = event_idx[g_start:g_end]
        d = idx_g.numel()                        
        sum_s_events = s[idx_g].sum()

        denom0 = cum_exp[idx_g[0]]
        exp_events = exp_s[idx_g].sum()
        denom_terms = 0.0
        for j in range(d):
            denom_terms += torch.log(denom0 - (j / d) * exp_events + 1e-12)

        pll = pll + (sum_s_events - denom_terms)
        start_ptr = g_end

    return pll

def cox_ph_loss(log_risk: torch.Tensor,
                durations: torch.Tensor,
                events: torch.Tensor,
                ties: str = 'efron',
                reduction: str = 'mean') -> torch.Tensor:
    events = events.float()
    pll = _efron_partial_ll(log_risk, durations, events)
    n_events = events.sum().clamp(min=1.0)
    loss = -(pll / n_events)
    if reduction == 'sum':
        return loss * n_events
    return loss
