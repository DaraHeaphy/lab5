"""NYU Depth v2 loading, preprocessing, splitting and (optional) joint augmentation."""
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms.v2 import functional as TF

IMG_SIZE = (192, 256)            # (H, W): keeps NYU's 3:4 aspect ratio, divisible by 32
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MAX_DEPTH = 10.0                 # NYU depth range in metres


# --------------------------------------------------------------------------- loading
def load_nyu(data_dir: str, cache_path: str | None = None, size=IMG_SIZE):
    """Load the whole NYU set, resized to `size`, as tensors.

    Returns (rgb uint8 [N,3,H,W], depth float32 [N,1,H,W] in metres).
    Supports the official `nyu_depth_v2_labeled.mat` or a folder of RGB/depth image pairs.
    The resized result is cached (a few hundred MB) so later sessions start instantly.
    """
    if cache_path and os.path.exists(cache_path):
        d = torch.load(cache_path)
        return d['rgb'], d['depth']

    mats = glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True)
    rgb, depth = _from_mat(mats[0], size) if mats else _from_images(data_dir, size)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        torch.save({'rgb': rgb, 'depth': depth}, cache_path)
    return rgb, depth


def _resize(rgb: torch.Tensor, depth: torch.Tensor, size):
    rgb = F.interpolate(rgb.float(), size=size, mode='bilinear', antialias=True, align_corners=False)
    depth = F.interpolate(depth.float(), size=size, mode='nearest')  # never blend depths across edges
    return rgb.round().clamp(0, 255).to(torch.uint8), depth


def _from_mat(path: str, size, chunk: int = 64):
    import h5py
    rgbs, depths = [], []
    with h5py.File(path, 'r') as f:
        n = f['images'].shape[0]
        for s in range(0, n, chunk):
            # MATLAB stores (N, C, W, H) -> swap to (N, C, H, W)
            im = torch.from_numpy(f['images'][s:s + chunk]).transpose(2, 3)
            dp = torch.from_numpy(f['depths'][s:s + chunk]).transpose(1, 2).unsqueeze(1)
            r, d = _resize(im, dp, size)
            rgbs.append(r), depths.append(d)
    return torch.cat(rgbs), torch.cat(depths)


def _from_images(data_dir: str, size):
    exts = ('.png', '.jpg', '.jpeg')
    files = [p for p in glob.glob(os.path.join(data_dir, '**', '*'), recursive=True)
             if p.lower().endswith(exts)]
    key = lambda p: os.path.splitext(p.lower())[0]
    depth_files = {key(p).replace('depth', ''): p for p in files if 'depth' in os.path.basename(p).lower()}
    rgb_files = {}
    for p in files:
        if 'depth' in os.path.basename(p).lower():
            continue
        k = key(p)
        for tok in ('rgb', 'image', 'colors', 'color'):
            k = k.replace(tok, '')
        rgb_files[k] = p
    common = sorted(set(depth_files) & set(rgb_files))
    if not common:
        raise RuntimeError(f'No RGB/depth pairs found under {data_dir}. '
                           'Expected nyu_depth_v2_labeled.mat or files with "depth" in their name.')

    rgbs, depths = [], []
    for k in common:
        im = torch.from_numpy(np.array(Image.open(rgb_files[k]).convert('RGB'))).permute(2, 0, 1)
        dp = np.array(Image.open(depth_files[k]))
        dp = dp / 1000.0 if dp.dtype == np.uint16 else dp * (MAX_DEPTH / 255.0)  # mm or 8-bit scaled
        r, d = _resize(im[None], torch.from_numpy(dp.astype(np.float32))[None, None], size)
        rgbs.append(r), depths.append(d)
    return torch.cat(rgbs), torch.cat(depths)


# --------------------------------------------------------------------------- splits
def make_splits(n: int, seed: int = 42, train=0.6, val=0.2, reduced=0.1, path: str | None = None):
    """Fixed 60/20/20 split plus a reduced train set (10% of the *whole* dataset, drawn from train).

    Saved to `path` so every model (and every Colab session) uses identical indices.
    """
    if path and os.path.exists(path):
        with open(path) as f:
            return {k: np.array(v) for k, v in json.load(f).items()}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr, n_val = int(train * n), int(val * n)
    splits = {
        'train': np.sort(perm[:n_tr]),
        'val': np.sort(perm[n_tr:n_tr + n_val]),
        'test': np.sort(perm[n_tr + n_val:]),
    }
    splits['reduced'] = np.sort(rng.choice(splits['train'], int(round(reduced * n)), replace=False))
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({k: v.tolist() for k, v in splits.items()}, f)
    return splits


# --------------------------------------------------------------------------- dataset
class NYUDepth(Dataset):
    """Serves (normalised RGB [3,H,W], depth in metres [1,H,W]) for the given indices."""

    def __init__(self, rgb, depth, indices, augment: bool = False):
        self.rgb, self.depth, self.idx, self.augment = rgb, depth, np.asarray(indices), augment
        self.jitter = ColorJitter(0.2, 0.2, 0.2, 0.05)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.rgb[j].float() / 255.0
        d = self.depth[j].clone()
        if self.augment:
            x, d = self._augment(x, d)
        return (x - IMAGENET_MEAN) / IMAGENET_STD, d

    def _augment(self, x, d):
        H, W = x.shape[-2:]
        if torch.rand(1) < 0.5:                                    # horizontal flip
            x, d = TF.horizontal_flip(x), TF.horizontal_flip(d)
        s = float(torch.empty(1).uniform_(0.75, 1.0))             # random crop, resized back
        h, w = int(H * s), int(W * s)
        top, left = int(torch.randint(0, H - h + 1, (1,))), int(torch.randint(0, W - w + 1, (1,)))
        x = TF.resized_crop(x, top, left, h, w, [H, W], antialias=True)
        d = TF.resized_crop(d, top, left, h, w, [H, W], interpolation=InterpolationMode.NEAREST)
        a = float(torch.empty(1).uniform_(-5, 5))                 # small rotation; padding gets depth 0 = masked
        x = TF.rotate(x, a, interpolation=InterpolationMode.BILINEAR)
        d = TF.rotate(d, a, interpolation=InterpolationMode.NEAREST, fill=0.0)
        x = self.jitter(x).clamp(0, 1)                             # colour jitter (RGB only)
        return x, d


def denormalise(x: torch.Tensor) -> torch.Tensor:
    return (x.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
