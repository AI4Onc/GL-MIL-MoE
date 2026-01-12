# run_all.py
import sys
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import json as _json

import pandas as pd
from lifelines.utils import concordance_index as _lifelines_cindex


ROOT = Path(__file__).resolve().parent 
PY = sys.executable


TRAIN_PY = ROOT / "src" / "train_cv.py"
INFER_PY = ROOT / "src" / "infer.py"
EVAL_PY  = ROOT / "src" / "evaluate.py"


EPOCHS = 25 
DEVICE = "cuda"
INFER_BATCH_SIZE = 128

GRID = [
    {"batch_size": 16, "lr": 3e-5, "weight_decay": 5e-5},
]

JSON_TRAIN = ROOT / "configs" / "CNN" / "dataset_train.json"
JSON_TEST  = ROOT / "configs" / "CNN" / "dataset_test.json"
JSON_VAL1  = ROOT / "configs" / "CNN" / "dataset_val.json"
JSON_VAL2  = ROOT / "configs" / "CNN" / "dataset_val2.json"

COHORTS = {
    "train": JSON_TRAIN,
    "test":  JSON_TEST,
    "val1":  JSON_VAL1,
    "val2":  JSON_VAL2,
}

SEED_BASE = 2025
USE_RUNTIME_SEED = True
N_FOLDS = 5


def tag_from_cfg(cfg: dict) -> str:
    # keep stable format
    return f"bs{cfg['batch_size']}_lr{cfg['lr']:g}_wd{cfg['weight_decay']:g}"

def runtime_dir_name(cfg: dict, runtime_i: int, seed: int, started: str) -> str:
    # your preferred naming style
    tag = tag_from_cfg(cfg)
    return f"{runtime_i}_{seed}_start_{started}"

def run(cmd, capture: bool = False, log_path: Path | None = None) -> str:
    """
    Run command in project root. If capture=True, capture stdout/stderr and optionally write to log.
    """
    cmd_str = " ".join(map(str, cmd))
    print("▶", cmd_str)

    if not capture:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        return ""

    p = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(out, encoding="utf-8", errors="ignore")

    if p.returncode != 0:
        raise RuntimeError(f"Command failed (rc={p.returncode}). See log: {log_path}\nCMD: {cmd_str}")
    return out

def parse_cindex(text: str) -> float:
    # match common formats from your evaluate.py
    m = re.search(r"C-index\)\s*:\s*([0-9]*\.[0-9]+|[0-9]+)", text)
    if m is None:
        m = re.search(r"C-index\s*:\s*([0-9]*\.[0-9]+|[0-9]+)", text)
    if m is None:
        raise ValueError("Failed to parse C-index from evaluate output.")
    return float(m.group(1))

def archive_ckpts(dst_dir: Path, n_folds: int = 5):
    """
    Copy checkpoints from experiments/checkpoints to dst_dir to prevent overwrite.
    """
    src_dir = ROOT / "experiments" / "checkpoints"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_folds):
        src = src_dir / f"bm_os_fold{i}.pt"
        if not src.exists():
            raise FileNotFoundError(f"Missing checkpoint: {src}")
        shutil.copy2(src, dst_dir / f"bm_os_fold{i}.pt")

def save_eval_df(pred_csv: Path, json_path: Path, out_path: Path) -> float:
    """
    Save per-cohort evaluation df:
    PatientID, duration, event, log_risk, score(=-log_risk)
    Return lifelines c-index on score (higher=longer survival).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    gt = {str(it["PatientID"]): (float(it["OS"]["Duration"]), int(it["OS"]["Event"])) for it in data}

    dfp = pd.read_csv(pred_csv)
    rows = []
    for _, r in dfp.iterrows():
        pid = str(r["PatientID"])
        if pid not in gt:
            continue
        dur, evt = gt[pid]
        log_risk = float(r["log_risk"])
        score = -log_risk
        rows.append({
            "PatientID": pid,
            "duration": dur,
            "event": evt,
            "log_risk": log_risk,
            "score": score,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfe = pd.DataFrame(rows)
    dfe.to_csv(out_path, index=False)

    if len(dfe) == 0:
        return float("nan")

    return float(_lifelines_cindex(dfe["duration"].values, dfe["score"].values, dfe["event"].values))

def write_run_meta(run_dir: Path, cfg: dict, runtime_i: int, started: str, seed: int):
    meta = {
        "run_tag": tag_from_cfg(cfg),
        "runtime": runtime_i,
        "train_batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "epochs": EPOCHS,
        "seed": seed,
        "device": DEVICE,
        "started_at": started,
    }
    (run_dir / "meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # sanity checks
    for p in [TRAIN_PY, INFER_PY, EVAL_PY]:
        if not p.exists():
            raise FileNotFoundError(f"Missing script: {p}")
    for name, jp in COHORTS.items():
        if not jp.exists():
            raise FileNotFoundError(f"Missing JSON ({name}): {jp}")

    #grid_root = ROOT / "experiments" / "grid_search"
    grid_root = ROOT / "Ablation" / "ResNet-18"
    grid_root.mkdir(parents=True, exist_ok=True)

    results = []

    for cfg in GRID:
        tag = tag_from_cfg(cfg)

        # you can change runtime count here
        for runtime_i in range(1, 2):
            seed = (SEED_BASE + runtime_i) if USE_RUNTIME_SEED else SEED_BASE
            started = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            run_dir = grid_root / tag / runtime_dir_name(cfg, runtime_i, seed, started)
            ckpt_dir = run_dir / "checkpoints"
            pred_dir = run_dir / "preds"
            eval_dir = run_dir / "eval"
            log_dir  = run_dir / "logs"

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            pred_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            write_run_meta(run_dir, cfg, runtime_i, started, seed)

            print("\n" + "=" * 90)
            print(f"🚀 Running config: {tag} | runtime={runtime_i} | epochs={EPOCHS} | seed={seed} | device={DEVICE}")
            print("=" * 90)

            # 1) Train CV
            train_cmd = [
                PY, str(TRAIN_PY),
                "--json", str(JSON_TRAIN),
                "--epochs", str(EPOCHS),
                "--batch_size", str(cfg["batch_size"]),
                "--lr", str(cfg["lr"]),
                "--weight_decay", str(cfg["weight_decay"]),
                "--device", DEVICE,
                "--seed", str(seed),
            ]
            run(train_cmd)

            # 2) Archive checkpoints
            archive_ckpts(ckpt_dir, n_folds=N_FOLDS)
            ckpts = [ckpt_dir / f"bm_os_fold{i}.pt" for i in range(N_FOLDS)]

            row = {
                "run_tag": tag,
                "runtime": runtime_i,
                "train_batch_size": cfg["batch_size"],
                "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
                "started_at": started,
                "seed": seed,
                "run_dir": str(run_dir),
            }

            # 3) Infer + Eval for each cohort
            for split, json_path in COHORTS.items():
                pred_csv = pred_dir / f"{split}_preds.csv"
                eval_log = log_dir / f"evaluate_{split}.log"

                try:
                    # infer
                    infer_cmd = [
                        PY, str(INFER_PY),
                        "--json", str(json_path),
                        "--ckpt", *[str(p) for p in ckpts],
                        "--ensemble", "risk_mean",
                        "--out_csv", str(pred_csv),
                        "--batch_size", str(INFER_BATCH_SIZE),
                        "--device", DEVICE
                    ]
                    run(infer_cmd)

                    # eval (capture output)
                    eval_cmd = [
                        PY, str(EVAL_PY),
                        "--pred_csv", str(pred_csv),
                        "--json", str(json_path),
                    ]
                    out = run(eval_cmd, capture=True, log_path=eval_log)

                    cidx = parse_cindex(out)
                    row[f"c_index_{split}"] = cidx

                    # save evaluation df
                    eval_df_path = eval_dir / f"{split}_eval_df.csv"
                    _ = save_eval_df(pred_csv, json_path, eval_df_path)

                except Exception as e:
                    row[f"c_index_{split}"] = None
                    row[f"error_{split}"] = str(e)

            print("✅", f"{tag} | runtime={runtime_i} | " +
                  " ".join([f"{k}={row.get('c_index_'+k)}" for k in COHORTS.keys()]))

            results.append(row)

            df = pd.DataFrame(results)
            out_xlsx = grid_root / "grid_search_results.xlsx"
            df.to_excel(out_xlsx, index=False)
            print(f"📄 Updated: {out_xlsx}")

    print("\n✅ All done.")