"""Training / evaluation loop: Huber loss, MARE metric, resumable runs saved to Drive."""
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models import DepthNet


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 2
    seed: int = 0


# --------------------------------------------------------------------------- loss & metric
def huber_loss(pred, gt, delta=1.0):
    valid = gt > 0                                           # 0 = missing / padded depth
    return F.huber_loss(pred[valid], gt[valid], delta=delta)


def mare_per_image(pred, gt):
    """Mean Absolute Relative Error |pred - gt| / gt over valid pixels, one value per image."""
    valid = (gt > 0).float()
    rel = (pred - gt).abs() / gt.clamp(min=1e-6) * valid
    return rel.flatten(1).sum(1) / valid.flatten(1).sum(1).clamp(min=1)


# --------------------------------------------------------------------------- training
def _lr_lambda(cfg, steps_per_epoch):
    warm, total = cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    return f


def train_model(model, train_ds, val_ds, cfg: TrainConfig, run_dir: str, device='cuda'):
    """Train with AdamW + warmup/cosine, keeping the best-on-validation weights.

    Everything is checkpointed to `run_dir` each epoch, so a Colab disconnect just resumes.
    Returns the history dict {'train_loss': [...], 'val_loss': [...], ...}.
    """
    os.makedirs(run_dir, exist_ok=True)
    hist_path, last_path = os.path.join(run_dir, 'history.json'), os.path.join(run_dir, 'last.pt')
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
        if hist.get('done'):
            return hist

    torch.manual_seed(cfg.seed)
    model.to(device)
    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=len(train_ds) > cfg.batch_size,
                          num_workers=cfg.num_workers, pin_memory=True, generator=g)
    val_dl = DataLoader(val_ds, cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg, len(train_dl)))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)
    hist = {'config': asdict(cfg), 'train_loss': [], 'val_loss': [], 'epoch_time': [], 'best_epoch': -1}
    start = 0
    if os.path.exists(last_path):                              # resume after a disconnect
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck['model']), opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched']), scaler.load_state_dict(ck['scaler'])
        hist, start = ck['hist'], ck['epoch'] + 1

    bar = tqdm(range(start, cfg.epochs), desc=os.path.basename(run_dir), initial=start, total=cfg.epochs)
    for epoch in bar:
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, d in train_dl:
            x, d = x.to(device, non_blocking=True), d.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.amp):
                pred = model(x)
            loss = huber_loss(pred.float(), d, cfg.huber_delta)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt), scaler.update(), sched.step()
            tot, n = tot + loss.item() * len(x), n + len(x)

        val = evaluate(model, val_dl, cfg, device)['huber']
        hist['train_loss'].append(tot / n)
        hist['val_loss'].append(val)
        hist['epoch_time'].append(time.time() - t0)
        if val <= min(hist['val_loss']):
            hist['best_epoch'] = epoch
            torch.save(model.state_dict(), os.path.join(run_dir, 'best.pt'))
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'scaler': scaler.state_dict(), 'hist': hist, 'epoch': epoch}, last_path)
        bar.set_postfix(train=f'{tot / n:.3f}', val=f'{val:.3f}')

    hist['done'] = True
    with open(hist_path, 'w') as f:
        json.dump(hist, f)
    os.remove(last_path)                                       # optimiser state no longer needed
    return hist


@torch.no_grad()
def evaluate(model, loader, cfg: TrainConfig, device='cuda'):
    model.eval()
    hub, n, mares = 0.0, 0, []
    for x, d in loader:
        x, d = x.to(device), d.to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.amp):
            pred = model(x)
        pred = pred.float()
        hub, n = hub + huber_loss(pred, d, cfg.huber_delta).item() * len(x), n + len(x)
        mares.append(mare_per_image(pred, d).cpu())
    mares = torch.cat(mares)
    return {'huber': hub / n, 'mare': mares.mean().item(), 'mare_std': mares.std().item()}


# --------------------------------------------------------------------------- experiment grid
def run_name(backbone, subset, augment):
    return f'{backbone}_{subset}' + ('_aug' if augment else '')


def run_experiment(backbone, subset, augment, datasets, cfg, out_dir, device='cuda'):
    """Train one (backbone, train-subset, augmentation) combination and score it on the test set.

    `datasets(subset, augment)` must return (train_ds, val_ds, test_ds).
    Results are cached in `out_dir/<run>/test.json`, so re-running the notebook is cheap.
    """
    run_dir = os.path.join(out_dir, run_name(backbone, subset, augment))
    test_path = os.path.join(run_dir, 'test.json')
    train_ds, val_ds, test_ds = datasets(subset, augment)
    model = DepthNet(backbone)
    hist = train_model(model, train_ds, val_ds, cfg, run_dir, device)
    if not os.path.exists(test_path):
        model.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'), map_location=device))
        model.to(device)
        res = evaluate(model, DataLoader(test_ds, cfg.batch_size, num_workers=cfg.num_workers), cfg, device)
        res.update(best_epoch=hist['best_epoch'], train_time_min=sum(hist['epoch_time']) / 60)
        with open(test_path, 'w') as f:
            json.dump(res, f)
    with open(test_path) as f:
        return hist, json.load(f)


def load_model(backbone, run_dir, device='cuda'):
    model = DepthNet(backbone)
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'), map_location=device))
    return model.to(device).eval()
