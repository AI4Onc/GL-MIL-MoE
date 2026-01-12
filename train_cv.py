from __future__ import annotations
import os
import argparse
import csv
import math
import time
import json
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from data.loading import load_dataset
from models.multimodal_cox import MultiModalCoxModel
from losses.cox import cox_ph_loss
from metrics.cindex import harrell_cindex


# -----------------------------
# Folds: event-aware if available; otherwise duration-only
# -----------------------------

def make_folds(items: Sequence[Dict], n_folds: int, seed: int = 42) -> List[List[int]]:
    def has_any_event(info: Sequence[Dict]) -> bool:
        for it in info:
            osf = it.get('OS')
            if isinstance(osf, dict) and osf.get('Event') is not None:
                return True
            if it.get('Event') is not None:
                return True
        return False

    def duration_of(i: int) -> float:
        osf = items[i].get('OS')
        if isinstance(osf, dict):
            return float(osf.get('Duration'))  
        if osf is not None:
            return float(osf) 
        return float(items[i].get('Duration')) 

    folds: List[List[int]] = [[] for _ in range(n_folds)]

    if has_any_event(items):
        idx_all = list(range(len(items)))
        evt_idx = [i for i in idx_all if (
            isinstance(items[i].get('OS'), dict) and int(items[i]['OS'].get('Event', items[i].get('Event', 0))) == 1
        ) or int(items[i].get('Event', 0)) == 1]
        cen_idx = [i for i in idx_all if i not in evt_idx]
        evt_idx.sort(key=duration_of)
        rng = random.Random(seed)
        rng.shuffle(cen_idx)
        for k, i in enumerate(evt_idx):
            folds[k % n_folds].append(i)
        for k, i in enumerate(cen_idx):
            folds[k % n_folds].append(i)
        return folds

    # No events: balance by duration
    idx = sorted(range(len(items)), key=duration_of)
    for k, i in enumerate(idx):
        folds[k % n_folds].append(i)
    return folds


# -----------------------------
# Train/eval helpers
# -----------------------------

def train_one_epoch(model: MultiModalCoxModel, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, scaler: torch.cuda.amp.GradScaler | None = None,
                    amp_dtype: torch.dtype | None = None) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_events = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        durations, events = batch['Duration'], batch['Event']
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and amp_dtype is not None:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                log_risk, _ = model(batch, return_details=True)
                loss = cox_ph_loss(log_risk, durations, events, ties='efron', reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            log_risk, _ = model(batch, return_details=True)
            loss = cox_ph_loss(log_risk, durations, events, ties='efron', reduction='mean')
            loss.backward()
            optimizer.step()

        batch_loss = float(loss.detach().item())
        batch_events = int(events.sum().item())
        total_loss += float(loss.detach().item()) * int(events.sum().item())
        total_events += int(events.sum().item())

    avg = total_loss / max(total_events, 1)
    return avg, total_events

def evaluate(model: MultiModalCoxModel, loader: DataLoader, device: torch.device,
             amp_dtype: torch.dtype | None = None) -> Tuple[float, float]:
    model.eval()
    all_loss = 0.0
    all_events = 0
    all_dur = []
    all_evt = []
    all_scores = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            durations, events = batch['Duration'], batch['Event']
            if amp_dtype is not None:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    out = model(batch, return_details=True)
            else:
                out = model(batch, return_details=True)
            log_risk = out[0] if isinstance(out, tuple) else out
            loss = cox_ph_loss(log_risk, durations, events, ties='efron', reduction='mean')

            all_loss += float(loss.detach().item()) * int(events.sum().item())
            all_events += int(events.sum().item())
            all_dur.append(durations)
            all_evt.append(events)
            all_scores.append(log_risk)
    avg_loss = all_loss / all_events if all_events > 0 else float('nan')
    durations = torch.cat(all_dur, dim=0)
    events = torch.cat(all_evt, dim=0)
    scores = torch.cat(all_scores, dim=0)
    cidx = harrell_cindex(durations, events, scores)
    return avg_loss, cidx

def main():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=str, required=True, help='Path to dataset JSON')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--weight_decay', type=float, default=5e-5)
    ap.add_argument('--amp', action='store_true', help='Use mixed precision (AMP)')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    timestamp = time.strftime('%m%d-%H%M')
    exp_name = f"{Path(args.json).stem}_seed{args.seed}_{timestamp}"

    """ckpt_dir = Path('experiments') / exp_name / 'checkpoints'
    runs_dir = Path('experiments') / exp_name / 'runs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)"""

    with open(args.json, 'r') as f:
        items = json.load(f)
    assert isinstance(items, list)

    folds = make_folds(items, n_folds=args.n_folds, seed=args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt_dir = Path('experiments/checkpoints'); ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path('experiments/runs'); runs_dir.mkdir(parents=True, exist_ok=True)

    oof_rows: List[Dict[str, str]] = []

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    amp_dtype = torch.float16 if args.amp else None

    for fold_idx in range(args.n_folds):
        val_idx = set(folds[fold_idx])
        train_idx = [i for k, fold in enumerate(folds) if k != fold_idx for i in fold]
        train_list = [items[i] for i in train_idx]
        val_list = [items[i] for i in val_idx]

        # 写成临时 JSON 文件再加载
        train_json = runs_dir / f"train_fold{fold_idx}.json"
        val_json = runs_dir / f"val_fold{fold_idx}.json"
        with open(train_json, 'w') as f: json.dump(train_list, f)
        with open(val_json, 'w') as f: json.dump(val_list, f)

        ds_tr = load_dataset(train_json)
        ds_va = load_dataset(val_json)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        model = MultiModalCoxModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_cidx = -math.inf
        best_state = None

        for epoch in range(1, args.epochs + 1):
            #tr_loss, tr_events = train_one_epoch(model, dl_tr, optimizer, device, amp_dtype=amp_dtype)
            tr_loss, tr_events = train_one_epoch(model, dl_tr, optimizer, device, scaler=scaler, amp_dtype=amp_dtype)
            va_loss, va_cidx = evaluate(model, dl_va, device, amp_dtype=amp_dtype)
            print(f"Fold {fold_idx} | Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}, val_cidx={va_cidx:.4f}")
            if not math.isnan(va_cidx) and va_cidx > best_cidx:
                best_cidx = va_cidx
                best_state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_cidx': va_cidx,
                }

        torch.save(best_state if best_state else {'model': model.state_dict()}, ckpt_dir / f"bm_os_fold{fold_idx}.pt")

        # OOF 推理
        model.load_state_dict(best_state['model'])
        model.to(device) 
        model.eval()
        with torch.no_grad():
            for batch in dl_va:
                pids = batch.get("PatientID", [None] * batch["Duration"].shape[0])
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                durations = batch["Duration"]
                events = batch["Event"]
                log_risk, _ = model(batch, return_details=True)
                for i in range(len(pids)):
                    oof_rows.append({
                        "PatientID": str(pids[i]),
                        "fold": str(fold_idx),
                        "duration": f"{float(durations[i].item()):.6f}",
                        "event": str(int(events[i].item())),
                        "log_risk": f"{float(log_risk[i].item()):.6f}",
                    })
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # OOF CSV
    oof_csv = runs_dir / 'oof_predictions.csv'
    with open(oof_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["PatientID", "fold", "duration", "event", "log_risk"])
        writer.writeheader()
        writer.writerows(oof_rows)

    d = torch.tensor([float(r["duration"]) for r in oof_rows], dtype=torch.float32)
    e = torch.tensor([int(r["event"]) for r in oof_rows], dtype=torch.float32)
    s = torch.tensor([float(r["log_risk"]) for r in oof_rows], dtype=torch.float32)
    cidx = harrell_cindex(d, e, s)
    print(f"OOF C-index: {cidx:.4f}")

if __name__ == '__main__':
    main()
