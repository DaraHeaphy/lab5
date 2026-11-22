"""Run the lab's training grid outside the notebook (same config as lab5.ipynb).

Usage:  python train_all.py [backbone ...] [--aug]
Finished runs are skipped and interrupted ones resume, so the notebook later just loads the results.
"""
import sys

from mypylibs.data import IMG_SIZE, NYUDepth, load_nyu, make_splits
from mypylibs.models import BACKBONES
from mypylibs.train import TrainConfig, run_experiment, run_name

DATA_DIR, OUT_DIR = 'data', 'runs'
CFG = TrainConfig(epochs=50, batch_size=16, lr=3e-4, weight_decay=0.05, warmup_epochs=3, huber_delta=1.0,
                  num_workers=0)

if __name__ == '__main__':
    args = sys.argv[1:]
    augment = '--aug' in args
    backbones = [a for a in args if a in BACKBONES] or list(BACKBONES)

    rgb, depth = load_nyu(DATA_DIR, cache_path=f'{OUT_DIR}/nyu_{IMG_SIZE[0]}x{IMG_SIZE[1]}.pt')
    splits = make_splits(len(rgb), seed=42, path=f'{OUT_DIR}/splits.json')

    def datasets(subset, aug=False):
        train_idx = splits['train'] if subset == 'full' else splits['reduced']
        return (NYUDepth(rgb, depth, train_idx, augment=aug), NYUDepth(rgb, depth, splits['val']),
                NYUDepth(rgb, depth, splits['test']))

    for subset in ('full', 'reduced'):
        for b in backbones:
            _, res = run_experiment(b, subset, augment, datasets, CFG, OUT_DIR)
            print(f"{run_name(b, subset, augment)}: test MARE {res['mare']:.4f} "
                  f"(best epoch {res['best_epoch'] + 1}, {res['train_time_min']:.1f} min)", flush=True)
