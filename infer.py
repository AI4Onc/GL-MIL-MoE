from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loading import load_dataset
from models.multimodal_cox import MultiModalCoxModel

def load_model(ckpt: str, device: torch.device) -> MultiModalCoxModel:
    model = MultiModalCoxModel().to(device)
    state = torch.load(ckpt, map_location=device)
    state_dict = state.get('model', state)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def load_baseline_npz(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    times = torch.tensor(data['times'], dtype=torch.float32)
    cumhaz = torch.tensor(data['cumhaz'], dtype=torch.float32)
    return times, cumhaz

def eval_baseline_at(times: torch.Tensor, cumhaz: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(times, t_query.clamp(min=times.min().item()), right=True) - 1
    idx = idx.clamp(min=0, max=len(cumhaz) - 1)
    return cumhaz[idx]

# -----------------------------
# Inference
# -----------------------------

@torch.no_grad()
def infer_once(model: MultiModalCoxModel, loader: DataLoader, device: torch.device) -> Tuple[List[str], torch.Tensor]:
    pids_all: List[str] = []
    scores: List[torch.Tensor] = []

    for batch in tqdm(loader, desc="🧠 Inference Progress", ncols=100):
        pids = batch.get('PatientID', [None] * batch['Duration'].shape[0])
        pids = [str(p) if p is not None else None for p in pids]
        pids_all.extend(pids)

        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model(batch)
        if isinstance(out, tuple):
            log_risk = out[0]
        else:
            log_risk = out
        scores.append(log_risk.detach().cpu())

    scores_cat = torch.cat(scores, dim=0) if scores else torch.empty(0)
    return pids_all, scores_cat

def main():
    ap = argparse.ArgumentParser(description='BM_OS inference (Cox)')
    ap.add_argument('--json', type=str, required=True)
    ap.add_argument('--ckpt', type=str, nargs='+', required=True, help='Checkpoint(s)')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--out_csv', type=str, default='experiments/runs/infer_preds.csv')
    ap.add_argument('--ensemble', type=str, default='risk_stack')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 🔁 使用你统一的加载函数
    ds = load_dataset(args.json)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 模型加载
    models = [load_model(p, device) for p in args.ckpt]

    # 推理
    all_pids = None
    per_model_scores = []

    for model in models:
        pids, scores = infer_once(model, dl, device)
        if all_pids is None:
            all_pids = pids
        per_model_scores.append(scores)

    assert all_pids is not None
    S = torch.stack(per_model_scores, dim=0)  # [M, N]

    if args.ensemble == 'risk_mean':
        agg_risk = S.mean(dim=0)
    elif args.ensemble == 'risk_max':
        agg_risk = S.max(dim=0).values
    elif args.ensemble == 'risk_min':
        agg_risk = S.min(dim=0).values
    elif args.ensemble == 'risk_median':
        agg_risk = S.median(dim=0).values
    elif args.ensemble == 'risk_stack':
        agg_risk = S.permute(1, 0)  # shape: [N, M]
    else:
        raise ValueError(f"Unsupported ensemble method: {args.ensemble}")
    # 写入 CSV
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', newline='') as f:
        if args.ensemble == 'risk_stack':
            fieldnames = ['PatientID'] + [f"log_risk_{i}" for i in range(S.shape[0])]
        else:
            fieldnames = ['PatientID', 'log_risk']

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, pid in enumerate(all_pids):
            row = {'PatientID': pid}
            if args.ensemble == 'risk_stack':
                for j in range(S.shape[0]):
                    row[f"log_risk_{j}"] = f"{float(S[j, i].item()):.6f}"
            else:
                row['log_risk'] = f"{float(agg_risk[i].item()):.6f}"
            writer.writerow(row)

    print(f"✅ Saved inference predictions to {outp}")

if __name__ == '__main__':
    main()
