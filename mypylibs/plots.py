"""Figures and tables for the report."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import denormalise
from .models import FAMILY


def show_samples(ds, n=4):
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5))
    for k, i in enumerate(np.linspace(0, len(ds) - 1, n).astype(int)):
        x, d = ds[i]
        axes[0, k].imshow(denormalise(x).permute(1, 2, 0))
        im = axes[1, k].imshow(d[0], cmap='magma_r', vmin=0, vmax=10)
        for a in axes[:, k]:
            a.axis('off')
    axes[0, 0].set_title('RGB input', loc='left'), axes[1, 0].set_title('Depth target (m)', loc='left')
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, label='depth (m)')
    plt.show()


def plot_loss_curves(histories: dict, backbones, subsets=('full', 'reduced'), suffix=''):
    """Grid: one row per training subset, one column per backbone; train vs. val Huber loss."""
    fig, axes = plt.subplots(len(subsets), len(backbones), figsize=(4 * len(backbones), 3.2 * len(subsets)),
                             sharey=True, squeeze=False)
    for r, s in enumerate(subsets):
        for c, b in enumerate(backbones):
            ax, h = axes[r, c], histories.get(f'{b}_{s}{suffix}')
            if h is None:
                ax.set_visible(False)
                continue
            ep = np.arange(1, len(h['train_loss']) + 1)
            ax.plot(ep, h['train_loss'], label='train')
            ax.plot(ep, h['val_loss'], label='validation')
            ax.axvline(h['best_epoch'] + 1, ls=':', c='grey', lw=1)
            ax.set_title(f'{b} ({FAMILY[b]}) - {s} set', fontsize=10)
            ax.set_xlabel('epoch'), ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel('Huber loss')
    axes[0, 0].legend()
    fig.tight_layout()
    plt.show()


def results_table(results: dict, backbones, subsets=('full', 'reduced'), suffix=''):
    """Test MARE per backbone x training subset, plus the relative degradation full -> reduced."""
    rows = []
    for b in backbones:
        row = {'model': b, 'family': FAMILY[b]}
        for s in subsets:
            r = results.get(f'{b}_{s}{suffix}')
            row[f'MARE {s}'] = r['mare'] if r else np.nan
        if len(subsets) == 2:
            row['degradation (%)'] = 100 * (row[f'MARE {subsets[1]}'] / row[f'MARE {subsets[0]}'] - 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index('model')


def plot_mare_bars(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith('MARE')]
    ax = df[cols].plot.bar(figsize=(7, 3.5), rot=0)
    ax.set_ylabel('test MARE (lower is better)'), ax.grid(axis='y', alpha=0.3)
    for cont in ax.containers:
        ax.bar_label(cont, fmt='%.3f', fontsize=8)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_predictions(models: dict, ds, indices, device='cuda'):
    """One row per test image: RGB, ground truth, then each model's prediction."""
    cols = 2 + len(models)
    fig, axes = plt.subplots(len(indices), cols, figsize=(2.6 * cols, 2.1 * len(indices)), squeeze=False)
    for r, i in enumerate(indices):
        x, d = ds[i]
        axes[r, 0].imshow(denormalise(x).permute(1, 2, 0))
        axes[r, 1].imshow(d[0], cmap='magma_r', vmin=0, vmax=10)
        for c, (name, m) in enumerate(models.items(), start=2):
            axes[r, c].imshow(m(x[None].to(device))[0, 0].cpu(), cmap='magma_r', vmin=0, vmax=10)
            if r == 0:
                axes[r, c].set_title(name, fontsize=8)
        for a in axes[r]:
            a.axis('off')
    axes[0, 0].set_title('RGB', fontsize=8), axes[0, 1].set_title('ground truth', fontsize=8)
    fig.tight_layout()
    plt.show()
