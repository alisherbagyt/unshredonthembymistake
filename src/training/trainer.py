# src/training/trainer.py
# eac-net training loop — infonce contrastive loss.
#
# dataset structure expected:
#   data/training_fragments/train/metadata.json   ← torn pairs, train split
#   data/training_fragments/val/metadata.json     ← torn pairs, val split
#   data/training_fragments/test/metadata.json    ← torn pairs, test split
#
#   data/DocLayNet/ is the SOURCE for synth_generator.py — it is NOT read here.
#   the trainer only reads pre-generated fragment pairs from training_fragments/.
#   DocLayNet content IS present in the training data because synth_generator
#   cut those pages to produce the pairs — it just isn't read directly again.
#
# key design decisions:
#   - fragment_pair_dataset caches geo features and cut_pts on first load
#     so compute_geometric_features() is called once per pair, not once per epoch
#   - strip extraction is vectorized (numpy broadcasting, no python loop)
#   - geo tensor is computed once and reused for both left and right sides
#     (they share the same cut boundary geometry by definition)
#   - drop_last=False on val/test so every pair contributes to accuracy
#   - best checkpoint = lowest VAL loss (not train loss)
#   - eac_net_last.pt stores optimizer + scheduler state for exact resume
#   - torch.compile on cuda for ~15-25% forward pass speedup (pytorch 2.x)
#   - mixed precision (torch.autocast) on cuda
#   - tqdm progress bars, training_log.csv, overfitting warning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json, csv, time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[trainer] tqdm not installed — run: pip install tqdm")

from src.features.eac_net  import build_model
from src.features.geometry import compute_geometric_features


# ── vectorized texture strip ──────────────────────────────────────────────────

def _extract_strip_fast(
    rgba:        np.ndarray,
    contour:     np.ndarray,
    strip_width: int,
) -> np.ndarray:
    """vectorized oriented strip extraction — single numpy advanced-index op.

    avoids all python loops. ~100x faster than the original nested loop.
    """
    rgb  = rgba[:, :, :3].astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    half = strip_width // 2

    prev = np.roll(contour,  1, axis=0)
    nxt  = np.roll(contour, -1, axis=0)
    tang = nxt - prev
    tang = tang / np.linalg.norm(tang, axis=1, keepdims=True).clip(1e-8)
    norm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)   # inward normal (n,2)

    offsets = np.arange(strip_width) - half               # (sw,)
    px = np.round(contour[:, 0:1] + offsets[None, :] * norm[:, 0:1]).astype(np.int32)
    py = np.round(contour[:, 1:2] + offsets[None, :] * norm[:, 1:2]).astype(np.int32)

    valid         = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    strip         = rgb[py.clip(0, h-1), px.clip(0, w-1)]   # (n, sw, 3)
    strip[~valid] = 0.0
    return strip.astype(np.float32)


# ── infonce loss ──────────────────────────────────────────────────────────────

class infonce_loss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = float(temperature)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        # emb_a, emb_b: (B*n, embed_dim) — already L2-normalized
        sim    = torch.mm(emb_a, emb_b.T) / self.tau   # (B*n, B*n)
        labels = torch.arange(len(emb_a), device=emb_a.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


# ── dataset ───────────────────────────────────────────────────────────────────

class fragment_pair_dataset(Dataset):
    """loads synthetic torn fragment pairs for eac-net training.

    performance notes:
      - geo features (compute_geometric_features) are cached on __init__
        because every pair has a fixed cut boundary — computing them in
        __getitem__ would redo identical work every epoch.
      - cut_pts are stored as np.float32 arrays, not raw lists.
      - strip extraction runs in __getitem__ because it depends on the
        image pixels which are not cached (too large to hold in RAM for
        4000+ pairs × 2 images × 1025×1025×3 bytes ≈ 24GB).
      - geo is the same for left and right (both sides share the cut boundary),
        so it's stored once per pair.
    """
    def __init__(self, metadata: list, strip_width: int = 16, n_pts: int = 512):
        self.strip_w = strip_width
        self.n_pts   = n_pts

        # pre-cache paths and geo features; filter out broken records
        self.records = []
        n_skipped = 0
        for m in metadata:
            left_path  = Path(m["left"])
            right_path = Path(m["right"])
            if not left_path.exists() or not right_path.exists():
                n_skipped += 1
                continue
            cut = np.array(m["cut_pts"], dtype=np.float32)   # (512, 2)
            geo = compute_geometric_features(cut)              # (512, 5) — cached
            self.records.append({
                "left":  str(left_path),
                "right": str(right_path),
                "cut":   cut,
                "geo":   geo,
            })
        if n_skipped:
            print(f"[dataset] skipped {n_skipped} pairs with missing image files")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]

        img_l = cv2.imread(r["left"],  cv2.IMREAD_COLOR)
        img_r = cv2.imread(r["right"], cv2.IMREAD_COLOR)

        z_geo = np.zeros((self.n_pts, 5),               dtype=np.float32)
        z_tex = np.zeros((self.n_pts, self.strip_w, 3), dtype=np.float32)

        if img_l is None or img_r is None:
            return {"geo": torch.from_numpy(z_geo),
                    "tex_l": torch.from_numpy(z_tex),
                    "tex_r": torch.from_numpy(z_tex)}

        # convert to rgba once each (alpha=255 means no masking needed here)
        rgba_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGBA)
        rgba_l[:, :, 3] = 255
        rgba_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGBA)
        rgba_r[:, :, 3] = 255

        tex_l = _extract_strip_fast(rgba_l, r["cut"], self.strip_w)
        tex_r = _extract_strip_fast(rgba_r, r["cut"], self.strip_w)

        # geo is shared for both sides — returned once, used twice in train loop
        return {
            "geo":   torch.from_numpy(r["geo"]),      # (512, 5)
            "tex_l": torch.from_numpy(tex_l),         # (512, sw, 3)
            "tex_r": torch.from_numpy(tex_r),         # (512, sw, 3)
        }


# ── metadata loading ──────────────────────────────────────────────────────────

def _load_split(out_dir: str, split: str) -> list:
    p = Path(out_dir) / split / "metadata.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    print(f"[trainer] {split:5s}: {len(data):5d} pairs  ← {p}")
    return data


# ── evaluation pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    use_amp:   bool,
    label:     str = "eval",
) -> dict:
    """loss + top-1 retrieval accuracy over a full dataloader.

    top-1 accuracy: for each contour point embedding ea[i], does argmax
    of cosine similarity with eb land on eb[i] (the correct match)?
    random chance = 1/512 ≈ 0.2%. well-trained model should exceed 50%+.

    drop_last=False on val/test loaders means partial final batches are
    included — no pairs are silently discarded from accuracy calculation.
    """
    model.eval()

    total_loss   = 0.0
    top1_correct = 0
    total_pts    = 0
    n_batches    = 0

    iter_loader = (
        tqdm(loader, desc=f"  {label}", unit="batch", leave=False,
             dynamic_ncols=True)
        if HAS_TQDM else loader
    )

    for batch in iter_loader:
        geo   = batch["geo"].to(device,   non_blocking=True)
        tex_l = batch["tex_l"].to(device, non_blocking=True)
        tex_r = batch["tex_r"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            emb_l = model(geo, tex_l)
            emb_r = model(geo, tex_r)
            b, n, d = emb_l.shape
            ea = emb_l.reshape(b * n, d)
            eb = emb_r.reshape(b * n, d)
            loss = criterion(ea, eb)

        total_loss   += loss.item()
        n_batches    += 1

        # retrieval accuracy (fp32 for numerical stability)
        sim            = torch.mm(ea.float(), eb.float().T)
        labels         = torch.arange(sim.shape[0], device=device)
        top1_correct  += (sim.argmax(dim=1) == labels).sum().item()
        total_pts     += sim.shape[0]

    return {
        "loss":     total_loss / max(1, n_batches),
        "accuracy": top1_correct / max(1, total_pts),
        "n_pts":    total_pts,
    }


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict):
    tc     = cfg["training"]
    fc     = cfg["features"]
    device = torch.device(fc.get("device", "cpu"))

    lr          = float(tc["lr"])
    temperature = float(tc["temperature"])
    epochs      = int(tc["epochs"])
    batch_size  = int(tc["batch_size"])
    strip_width = int(fc["strip_width"])
    accum_steps = int(tc.get("accum_steps", 4))

    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    out_dir = tc["out_dir"]

    # ── load splits ───────────────────────────────────────────────────────────
    print(f"[trainer] loading splits from {out_dir}/")
    train_meta = _load_split(out_dir, "train")
    val_meta   = _load_split(out_dir, "val")
    test_meta  = _load_split(out_dir, "test")

    if not train_meta:
        raise FileNotFoundError(
            f"no train metadata at {out_dir}/train/metadata.json\n"
            "run: python train.py --generate-data --doclayet-root data/DocLayNet"
        )

    # ── datasets & loaders ───────────────────────────────────────────────────
    print("[trainer] caching geometric features (one-time, ~10s)...")
    t_cache = time.perf_counter()
    train_ds = fragment_pair_dataset(train_meta, strip_width)
    val_ds   = fragment_pair_dataset(val_meta,   strip_width) if val_meta   else None
    test_ds  = fragment_pair_dataset(test_meta,  strip_width) if test_meta  else None
    print(f"[trainer] feature cache built in {time.perf_counter()-t_cache:.1f}s")

    def make_loader(ds, shuffle, drop_last):
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = 0,           # must be 0 on windows + cuda
            pin_memory  = (device.type == "cuda"),
            drop_last   = drop_last,   # True for train (uniform batches), False for eval
        )

    train_loader = make_loader(train_ds, shuffle=True,  drop_last=True)
    val_loader   = make_loader(val_ds,   shuffle=False, drop_last=False) if val_ds   else None
    test_loader  = make_loader(test_ds,  shuffle=False, drop_last=False) if test_ds  else None

    # ── model ─────────────────────────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = infonce_loss(temperature)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # torch.compile: free ~15-25% throughput on cuda (pytorch 2.x)
    # requires Triton, which is Linux-only — skip silently on Windows
    import platform
    if device.type == "cuda" and platform.system() != "Windows":
        try:
            model = torch.compile(model)
            print("[trainer] torch.compile enabled")
        except Exception:
            print("[trainer] torch.compile skipped (unsupported environment)")
    else:
        print(f"[trainer] torch.compile skipped "
              f"({'Windows — Triton not available' if platform.system() == 'Windows' else 'cpu'})")

    ckpt_dir  = Path(tc["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "eac_net_best.pt"
    last_path = ckpt_dir / "eac_net_last.pt"
    log_path  = ckpt_dir / "training_log.csv"

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")

    if last_path.exists():
        print(f"[trainer] resuming from {last_path}")
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        # handle compiled model (state dict keys may have '_orig_mod.' prefix)
        state = ckpt["model"]
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # strip compile prefix if checkpoint was saved from non-compiled model
            from collections import OrderedDict
            new_state = OrderedDict(
                (k.replace("_orig_mod.", ""), v) for k, v in state.items()
            )
            model.load_state_dict(new_state, strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[trainer] resuming at epoch {start_epoch}, "
              f"best val loss: {best_val_loss:.4f}")

    # ── log ───────────────────────────────────────────────────────────────────
    write_header = not log_path.exists()
    log_file     = open(log_path, "a", newline="")
    log_writer   = csv.writer(log_file)
    if write_header:
        log_writer.writerow([
            "epoch", "train_loss", "val_loss", "val_acc",
            "best_val_loss", "lr", "elapsed_s"
        ])

    n_params = sum(p.numel() for p in model.parameters())
    eff_batch = batch_size * accum_steps
    print(f"\n[trainer] ── config ──────────────────────────────────────────")
    print(f"  device         : {device}")
    print(f"  amp            : {'on (fp16)' if use_amp else 'off (fp32)'}")
    print(f"  batch / accum  : {batch_size} × {accum_steps} = {eff_batch} effective")
    print(f"  lr / epochs    : {lr}  /  {start_epoch}→{epochs}")
    print(f"  train pairs    : {len(train_ds)}")
    print(f"  val   pairs    : {len(val_ds)   if val_ds   else 0}")
    print(f"  test  pairs    : {len(test_ds)  if test_ds  else 0}")
    print(f"  model params   : {n_params:,}")
    print(f"  best ckpt      : {best_path}  (best val loss)")
    print(f"  last ckpt      : {last_path}  (every epoch, resumable)")
    print(f"[trainer] ────────────────────────────────────────────────────\n")

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        t0         = time.perf_counter()
        optimizer.zero_grad()

        iter_loader = (
            tqdm(train_loader,
                 desc=f"epoch {epoch:03d}/{epochs}",
                 unit="batch", leave=False, dynamic_ncols=True)
            if HAS_TQDM else train_loader
        )

        for step, batch in enumerate(iter_loader):
            geo   = batch["geo"].to(device,   non_blocking=True)
            tex_l = batch["tex_l"].to(device, non_blocking=True)
            tex_r = batch["tex_r"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                emb_l = model(geo, tex_l)   # geo shared for both sides
                emb_r = model(geo, tex_r)
                b, n, d = emb_l.shape
                loss = criterion(
                    emb_l.reshape(b * n, d),
                    emb_r.reshape(b * n, d),
                ) / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * accum_steps
            n_batches  += 1

            if (step + 1) % accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            if HAS_TQDM:
                iter_loader.set_postfix(
                    loss=f"{loss.item()*accum_steps:.4f}",
                    best_val=f"{best_val_loss:.4f}",
                )

        # flush any remaining accumulated gradients
        if n_batches % accum_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        train_loss = total_loss / max(1, n_batches)
        elapsed    = time.perf_counter() - t0
        cur_lr     = scheduler.get_last_lr()[0]

        # ── validate ──────────────────────────────────────────────────────────
        val_loss = float("nan")
        val_acc  = float("nan")
        if val_loader:
            val_m    = _evaluate(model, val_loader, criterion, device,
                                 use_amp, label="val")
            val_loss = val_m["loss"]
            val_acc  = val_m["accuracy"]

        # ── checkpoint — best = lowest val loss ───────────────────────────────
        monitor = val_loss if val_loader else train_loss
        is_best = (monitor == monitor) and (monitor < best_val_loss)  # nan-safe

        # always save last (full state for resume)
        torch.save({
            "epoch":         epoch,
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "val_accuracy":  val_acc,
            "best_val_loss": best_val_loss,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
        }, last_path)

        if is_best:
            best_val_loss = monitor
            torch.save({
                "epoch":       epoch,
                "train_loss":  train_loss,
                "val_loss":    val_loss,
                "val_accuracy": val_acc,
                "model":       model.state_dict(),
            }, best_path)

        marker = "  ← best" if is_best else ""

        if val_loader:
            print(f"[trainer] {epoch:03d}/{epochs}  "
                  f"train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  acc={val_acc:.1%}  "
                  f"lr={cur_lr:.2e}  {elapsed:.0f}s{marker}")
            # overfitting warning after epoch 10
            if epoch >= 10 and (val_loss - train_loss) > 0.5:
                print(f"[trainer] WARNING: val−train gap = "
                      f"{val_loss-train_loss:.3f} — possible overfitting")
        else:
            print(f"[trainer] {epoch:03d}/{epochs}  "
                  f"train={train_loss:.4f}  "
                  f"lr={cur_lr:.2e}  {elapsed:.0f}s{marker}")

        log_writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{best_val_loss:.6f}",
            f"{cur_lr:.2e}",
            f"{elapsed:.1f}",
        ])
        log_file.flush()

    log_file.close()

    # ── final test evaluation using best checkpoint ───────────────────────────
    if test_loader and best_path.exists():
        print(f"\n[trainer] final test evaluation (best model)...")
        ckpt      = torch.load(best_path, map_location=device, weights_only=True)
        state     = ckpt["model"]
        try:
            model.load_state_dict(state)
        except RuntimeError:
            from collections import OrderedDict
            model.load_state_dict(
                OrderedDict((k.replace("_orig_mod.", ""), v)
                            for k, v in state.items()), strict=False)
        test_m = _evaluate(model, test_loader, criterion, device,
                           use_amp, label="test")
        print(f"[trainer] test loss     : {test_m['loss']:.4f}")
        print(f"[trainer] test accuracy : {test_m['accuracy']:.2%}  "
              f"({int(test_m['accuracy']*test_m['n_pts'])}"
              f" / {test_m['n_pts']} points)")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                "TEST", "", "", "",
                f"test_loss={test_m['loss']:.6f}",
                f"test_acc={test_m['accuracy']:.6f}", "",
            ])

    print(f"\n[trainer] ── done ────────────────────────────────────────────")
    print(f"  best val loss  : {best_val_loss:.4f}")
    print(f"  best weights   : {best_path}")
    print(f"  last weights   : {last_path}")
    print(f"  log            : {log_path}")
    print(f"\n  next — set in configs/config.yaml:")
    print(f'    features.weights_path: "checkpoints/eac_net_best.pt"')
    print(f'    features.device: "cuda"')