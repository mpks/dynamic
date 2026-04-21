"""
patterson_cnn.py

3D residual filter CNN that learns to correct observed Patterson maps
(dynamically distorted) towards kinematic (calculated) Patterson maps.

Architecture: 3D patch-based residual network
  - Operates on small 3D patches extracted from .patt volumes
  - No downsampling — acts as a pure spatial filter
  - Residual learning: network predicts the CORRECTION (obs - cal),
    not the full map. At init (weights≈0) → identity transform.
  - Patch-based training gives many samples per dataset → transferable

Data flow:
  Training:
    obs.patt + cal.patt  →  extract patches  →  train network
    target = obs_patch - cal_patch  (the distortion to remove)
    loss = Patterson-space loss + reciprocal-space loss (R-factor)

  Inference:
    obs.patt  →  sliding-window correction  →  corrected.patt
    corrected.patt  →  3D FFT  →  corrected |F_hkl|²
    →  R1 evaluation and Fc vs Fo_corrected plot

Usage:
    from patterson_cnn import PattersonFilterCNN, PattersonTrainer
    from patterson_cnn import PattersonPatchDataset, correct_volume
    from patterson_cnn import back_transform_3d, evaluate_correction

    # Training
    dataset = PattersonPatchDataset.from_patt_pairs(
        [('obs1.patt', 'cal1.patt'), ('obs2.patt', 'cal2.patt')],
        patch_size=16, stride=4
    )
    train_ds, val_ds = dataset.split(val_fraction=0.15)
    model = PattersonFilterCNN(patch_size=16, base_ch=32, n_blocks=8)
    trainer = PattersonTrainer(model)
    trainer.train(train_ds, val_ds, n_epochs=100)

    # Inference
    corrected_map = correct_volume('new_obs.patt', model)
    hkl_corrections = back_transform_3d(corrected_map, spots_list)
    evaluate_correction(hkl_corrections, spots_list)
"""

import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import matplotlib.pyplot as plt


# ── .patt file I/O ──────────────────────────────────────────────────────────

def read_patt(path: str) -> Tuple[np.ndarray, str]:
    """Read a .patt file. Returns (grid, label) where grid is (nx,ny,nz) float32."""
    with open(path, 'rb') as f:
        buf = f.read()
    hdr_len = struct.unpack_from('<I', buf, 0)[0]
    hdr = buf[4:4+hdr_len].decode('ascii').strip().split('\n')
    if hdr[0].strip() != 'PATT1':
        raise ValueError(f"{path}: not a valid .patt file")
    nx, ny, nz = map(int, hdr[1].strip().split())
    label = hdr[2].strip() if len(hdr) > 2 else 'unknown'
    data = np.frombuffer(buf, dtype='<f4', offset=4+hdr_len,
                         count=nx*ny*nz).copy().reshape(nx, ny, nz)
    return data, label


def write_patt(grid: np.ndarray, path: str, label: str = 'corrected') -> None:
    """Write a .patt file from (nx,ny,nz) float32 array."""
    nx, ny, nz = grid.shape
    header = f"PATT1\n{nx} {ny} {nz}\n{label}\n".encode('ascii')
    pad = (-(4 + len(header))) % 4
    header += b'\x00' * pad
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(header)))
        f.write(header)
        f.write(grid.astype('<f4').tobytes())


# ── Data container ───────────────────────────────────────────────────────────

@dataclass
class PattPair:
    """
    One (obs, cal) Patterson map pair from a single dataset.
    obs and cal are (nx, ny, nz) float32 arrays, normalised to [0,1].
    dataset_id is a string label for bookkeeping.
    """
    obs:        np.ndarray   # (nx, ny, nz) observed Patterson
    cal:        np.ndarray   # (nx, ny, nz) calculated (kinematic) Patterson
    dataset_id: str

    @classmethod
    def from_patt_files(cls, obs_path: str, cal_path: str) -> 'PattPair':
        obs, lbl_obs = read_patt(obs_path)
        cal, lbl_cal = read_patt(cal_path)
        if obs.shape != cal.shape:
            raise ValueError(
                f"Shape mismatch: obs={obs.shape} cal={cal.shape}. "
                f"Recompute both maps with the same grid_size."
            )
        # Normalise each to [0,1] independently
        obs = _norm01(obs)
        cal = _norm01(cal)
        dataset_id = Path(obs_path).stem.replace('_obs', '')
        return cls(obs=obs, cal=cal, dataset_id=dataset_id)


def _norm01(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    if mx > mn:
        return (x - mn) / (mx - mn)
    return x - mn


# ── Patch dataset ─────────────────────────────────────────────────────────────

class PattersonPatchDataset(Dataset):
    """
    Extracts overlapping 3D patches from (obs, cal) Patterson volume pairs.

    Each patch is an independent training sample. A single 64³ volume with
    patch_size=16 and stride=8 yields (64-16)//8 + 1)³ = 7³ = 343 patches.
    With stride=4: 13³ = 2197 patches per dataset.

    The network learns to predict the CORRECTION:
        target = obs_patch - cal_patch    (what needs to be subtracted)
    At inference: cal_predicted = obs - network(obs)

    Augmentation exploits Patterson symmetry:
        P(u,v,w) = P(-u,-v,-w)   → inversion: flip all three axes simultaneously
        P(u,v,w) = P(-u,v,w) etc → individual axis flips (for orthorhombic/higher)
    For a general (triclinic) dataset only the inversion is guaranteed,
    so we use all 2³=8 combinations of axis flips (all valid by inversion symmetry
    applied to pairs of axes) plus the full inversion.
    """

    def __init__(self,
                 patches_obs: np.ndarray,   # (N, P, P, P)
                 patches_cal: np.ndarray,   # (N, P, P, P)
                 dataset_ids: List[str],
                 augment: bool = True):
        assert len(patches_obs) == len(patches_cal)
        self.obs        = patches_obs
        self.cal        = patches_cal
        self.ids        = dataset_ids
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        obs = self.obs[idx].copy()   # (P, P, P)
        cal = self.cal[idx].copy()

        if self.augment:
            # Flip each axis independently — all valid by Patterson inversion symmetry
            for ax in range(3):
                if np.random.rand() > 0.5:
                    obs = np.flip(obs, axis=ax).copy()
                    cal = np.flip(cal, axis=ax).copy()
            # Random 90° rotations in each plane
            for ax0, ax1 in [(0,1), (0,2), (1,2)]:
                k = np.random.randint(0, 4)
                obs = np.rot90(obs, k=k, axes=(ax0, ax1)).copy()
                cal = np.rot90(cal, k=k, axes=(ax0, ax1)).copy()

        x      = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1,P,P,P)
        target = torch.tensor(obs - cal, dtype=torch.float32).unsqueeze(0)  # correction
        return x, target

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def from_patt_pairs(cls,
                        pairs: List[Tuple[str, str]],
                        patch_size: int = 16,
                        stride: int = 8,
                        augment: bool = True) -> 'PattersonPatchDataset':
        """
        Build dataset from a list of (obs_path, cal_path) tuples.

        Parameters
        ----------
        pairs      : list of (obs.patt, cal.patt) file path pairs
        patch_size : size of cubic patches (P)
        stride     : stride between patch centres
        augment    : apply data augmentation
        """
        all_obs, all_cal, all_ids = [], [], []

        for obs_path, cal_path in pairs:
            pair = PattPair.from_patt_files(obs_path, cal_path)
            obs_patches, cal_patches = _extract_patches(
                pair.obs, pair.cal, patch_size, stride
            )
            n = len(obs_patches)
            all_obs.append(obs_patches)
            all_cal.append(cal_patches)
            all_ids.extend([pair.dataset_id] * n)
            print(f"  {pair.dataset_id}: {pair.obs.shape} → {n} patches")

        obs_arr = np.concatenate(all_obs, axis=0)
        cal_arr = np.concatenate(all_cal, axis=0)
        print(f"Total patches: {len(obs_arr)}")
        return cls(obs_arr, cal_arr, all_ids, augment=augment)

    def split(self,
              val_fraction: float = 0.15,
              seed: int = 42) -> Tuple['PattersonPatchDataset',
                                       'PattersonPatchDataset']:
        """
        Split into train/val sets.
        Split is done at the DATASET level (not patch level) to avoid
        data leakage: all patches from one volume stay together.
        """
        rng = np.random.default_rng(seed)

        # Find unique dataset ids
        unique_ids = list(dict.fromkeys(self.ids))  # preserves order
        rng.shuffle(unique_ids)
        n_val = max(1, min(int(len(unique_ids) * val_fraction),
                           len(unique_ids) - 1))
        val_ids  = set(unique_ids[:n_val])
        train_ids = set(unique_ids[n_val:])

        train_mask = np.array([id_ in train_ids for id_ in self.ids])
        val_mask   = ~train_mask

        def subset(mask, aug):
            return PattersonPatchDataset(
                self.obs[mask], self.cal[mask],
                [id_ for id_, m in zip(self.ids, mask) if m],
                aug
            )

        train_ds = subset(train_mask, True)
        val_ds   = subset(val_mask,   False)
        print(f"Train patches: {len(train_ds)}  |  Val patches: {len(val_ds)}")
        return train_ds, val_ds


def _extract_patches(obs: np.ndarray,
                     cal: np.ndarray,
                     patch_size: int,
                     stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract all overlapping cubic patches from two aligned volumes."""
    P = patch_size
    nx, ny, nz = obs.shape
    obs_patches, cal_patches = [], []
    for i in range(0, nx - P + 1, stride):
        for j in range(0, ny - P + 1, stride):
            for k in range(0, nz - P + 1, stride):
                obs_patches.append(obs[i:i+P, j:j+P, k:k+P])
                cal_patches.append(cal[i:i+P, j:j+P, k:k+P])
    return np.array(obs_patches), np.array(cal_patches)


# ── Model ─────────────────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    """
    3D residual block: Conv3D → BN → ReLU → Conv3D → BN + skip.
    Uses 3×3×3 kernels throughout — purely local filter.
    No downsampling — preserves spatial resolution exactly.
    """
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class PattersonFilterCNN(nn.Module):
    """
    3D residual filter that maps an observed Patterson map patch to
    the correction that should be subtracted from it.

    At initialisation all weights are small → network predicts ≈0 correction
    → identity transform. Training nudges it to learn the distortion.

    Architecture:
        input (1, P, P, P)
            ↓ expand to base_ch channels
        [ResBlock3D] × n_blocks   ← purely local 3×3×3 filter
            ↓ collapse to 1 channel
        output (1, P, P, P)       ← predicted correction

    Full volume correction (inference):
        corrected[u,v,w] = obs[u,v,w] - network(obs)[u,v,w]
        implemented as sliding window with overlap-average blending.

    Parameters
    ----------
    patch_size : spatial size of input patches (not used in forward,
                 stored for reference)
    base_ch    : number of channels in residual blocks
    n_blocks   : depth of the filter (more blocks = larger receptive field)
                 receptive field = 1 + n_blocks * 2 * (kernel_size - 1)
                                 = 1 + n_blocks * 4  (for 3×3×3 kernels)
                 n_blocks=8  → receptive field = 33 voxels
                 n_blocks=12 → receptive field = 49 voxels
    dropout    : dropout rate inside residual blocks (regularisation)
    """

    def __init__(self,
                 patch_size: int = 16,
                 base_ch:    int = 32,
                 n_blocks:   int = 8,
                 dropout:    float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.base_ch    = base_ch
        self.n_blocks   = n_blocks

        # Input projection: 1 → base_ch
        self.input_proj = nn.Sequential(
            nn.Conv3d(1, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
        )

        # Residual filter stack
        self.blocks = nn.Sequential(
            *[ResBlock3D(base_ch, dropout=dropout) for _ in range(n_blocks)]
        )

        # Output projection: base_ch → 1
        # Initialise final conv weights to near-zero so network starts
        # as identity (predicts zero correction)
        self.output_proj = nn.Conv3d(base_ch, 1, kernel_size=1)
        nn.init.normal_(self.output_proj.weight, std=1e-3)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, P, P, P) observed Patterson patch

        Returns
        -------
        correction : (B, 1, P, P, P)
            Predicted correction. Apply as: cal_predicted = x - correction
        """
        h = self.input_proj(x)
        h = self.blocks(h)
        correction = self.output_proj(h)
        return correction

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Reciprocal-space loss ──────────────────────────────────────────────────────

def reciprocal_space_loss(pred_cal: torch.Tensor,
                          true_cal: torch.Tensor) -> torch.Tensor:
    """
    R-factor-inspired loss in reciprocal space.

    Takes the 3D FFT of both maps and compares the amplitude spectra.
    This penalises errors in the actual structure factor amplitudes,
    not just the Patterson map pixel values.

    pred_cal, true_cal : (B, 1, P, P, P) Patterson map patches
    """
    # FFT of each patch — rfft3 gives the positive-frequency half
    F_pred = torch.fft.rfftn(pred_cal, dim=(-3,-2,-1))
    F_true = torch.fft.rfftn(true_cal, dim=(-3,-2,-1))

    amp_pred = torch.abs(F_pred)
    amp_true = torch.abs(F_true)

    # Normalise amplitudes — we care about relative not absolute values
    amp_pred = amp_pred / (amp_pred.mean(dim=(-3,-2,-1), keepdim=True) + 1e-8)
    amp_true = amp_true / (amp_true.mean(dim=(-3,-2,-1), keepdim=True) + 1e-8)

    return F.l1_loss(amp_pred, amp_true)


# ── Trainer ───────────────────────────────────────────────────────────────────

class PattersonTrainer:
    """
    Trains PattersonFilterCNN with a combined loss:
        loss = α * L_patterson + β * L_reciprocal

    L_patterson  : L1 + MSE on corrected Patterson patches
    L_reciprocal : L1 on FFT amplitude spectra (structure factor amplitudes)

    The reciprocal-space loss directly optimises what we care about
    (corrected |F_hkl|) and is complementary to the map-space loss.
    """

    def __init__(self,
                 model:        PattersonFilterCNN,
                 device:       str   = 'cuda',
                 lr:           float = 1e-3,
                 weight_decay: float = 1e-4,
                 alpha:        float = 0.7,   # weight for Patterson loss
                 beta:         float = 0.3):  # weight for reciprocal loss
        self.model  = model.to(device)
        self.device = device
        self.alpha  = alpha
        self.beta   = beta

        self.opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=100, eta_min=1e-5
        )
        self.history = {'train': [], 'val': [], 'train_rspace': [], 'val_rspace': []}

        print(f"Model parameters: {model.n_parameters():,}")
        print(f"Device: {device}")

    def _compute_loss(self,
                      x: torch.Tensor,
                      target_correction: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x                  : (B,1,P,P,P) observed patch
        target_correction  : (B,1,P,P,P) = obs - cal  (what to subtract)

        Returns (total_loss, reciprocal_loss)
        """
        pred_correction = self.model(x)

        # Predicted corrected map
        pred_cal = x - pred_correction
        true_cal = x - target_correction

        # Patterson-space loss on the correction itself
        l1  = F.l1_loss(pred_correction, target_correction)
        mse = F.mse_loss(pred_correction, target_correction)
        l_patt = 0.5 * l1 + 0.5 * mse

        # Reciprocal-space loss on the resulting corrected maps
        l_rspace = reciprocal_space_loss(pred_cal, true_cal)

        total = self.alpha * l_patt + self.beta * l_rspace
        return total, l_rspace

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total, rspace = 0.0, 0.0
        for x, target in loader:
            x      = x.to(self.device)
            target = target.to(self.device)
            self.opt.zero_grad()
            loss, lr = self._compute_loss(x, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total  += loss.item()
            rspace += lr.item()
        n = len(loader)
        return total / n, rspace / n

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total, rspace = 0.0, 0.0
        for x, target in loader:
            x      = x.to(self.device)
            target = target.to(self.device)
            loss, lr = self._compute_loss(x, target)
            total  += loss.item()
            rspace += lr.item()
        n = len(loader)
        return total / n, rspace / n

    def train(self,
              train_ds:        PattersonPatchDataset,
              val_ds:          PattersonPatchDataset,
              n_epochs:        int  = 100,
              batch_size:      int  = 16,
              num_workers:     int  = 4,
              checkpoint_path: str  = 'patterson_filter.pt'):

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        best_val = float('inf')

        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_rsp = self.train_epoch(train_loader)
            va_loss, va_rsp = self.val_epoch(val_loader)
            self.scheduler.step()

            self.history['train'].append(tr_loss)
            self.history['val'].append(va_loss)
            self.history['train_rspace'].append(tr_rsp)
            self.history['val_rspace'].append(va_rsp)

            print(f"Epoch {epoch:04d}  "
                  f"train {tr_loss:.5f} (rsp {tr_rsp:.5f})  "
                  f"val {va_loss:.5f} (rsp {va_rsp:.5f})")

            if va_loss < best_val:
                best_val = va_loss
                self.save(checkpoint_path, epoch, va_loss)
                print(f"  → checkpoint saved (val {best_val:.5f})")

    def save(self, path: str, epoch: int, val_loss: float) -> None:
        torch.save({
            'epoch':       epoch,
            'model_state': self.model.state_dict(),
            'opt_state':   self.opt.state_dict(),
            'val_loss':    val_loss,
            'patch_size':  self.model.patch_size,
            'base_ch':     self.model.base_ch,
            'n_blocks':    self.model.n_blocks,
            'history':     self.history,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'PattersonFilterCNN':
        """Load a saved model. Returns the model (not the trainer)."""
        ckpt = torch.load(path, map_location=device)
        model = PattersonFilterCNN(
            patch_size = ckpt['patch_size'],
            base_ch    = ckpt['base_ch'],
            n_blocks   = ckpt['n_blocks'],
        )
        model.load_state_dict(ckpt['model_state'])
        model.to(device)
        model.eval()
        print(f"Loaded model from epoch {ckpt['epoch']} "
              f"(val loss {ckpt['val_loss']:.5f})")
        return model

    def plot_losses(self, filename: Optional[str] = None) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, key, title in zip(
            axes,
            [('train','val'), ('train_rspace','val_rspace')],
            ['Patterson-space loss', 'Reciprocal-space loss']
        ):
            ax.plot(self.history[key[0]], label='Train')
            ax.plot(self.history[key[1]], label='Val')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title(title); ax.legend()
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=200)
        else:
            plt.show()


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def correct_volume(obs_path:   str,
                   model:      PattersonFilterCNN,
                   patch_size: int = 16,
                   stride:     int = 8,
                   device:     str = 'cuda',
                   out_path:   Optional[str] = None) -> np.ndarray:
    """
    Correct a full observed Patterson volume using sliding-window inference.

    Overlapping patches are averaged using a Gaussian weight window to
    avoid boundary artefacts at patch edges.

    Parameters
    ----------
    obs_path   : path to observed .patt file
    model      : trained PattersonFilterCNN
    patch_size : must match training patch size
    stride     : sliding step (smaller = more overlap = smoother but slower)
    device     : 'cuda' or 'cpu'
    out_path   : if given, write corrected map to this .patt file

    Returns
    -------
    corrected : (nx, ny, nz) float32 corrected Patterson map, normalised [0,1]
    """
    model.eval()
    model.to(device)

    obs, label = read_patt(obs_path)
    obs = _norm01(obs)
    nx, ny, nz = obs.shape
    P = patch_size

    # Gaussian window for smooth blending at patch boundaries
    g1d = np.exp(-0.5 * ((np.arange(P) - (P-1)/2) / (P/4))**2)
    window = g1d[:,None,None] * g1d[None,:,None] * g1d[None,None,:]
    window = window.astype(np.float32)

    correction_sum  = np.zeros_like(obs)
    weight_sum      = np.zeros_like(obs)

    for i in range(0, nx - P + 1, stride):
        for j in range(0, ny - P + 1, stride):
            for k in range(0, nz - P + 1, stride):
                patch = obs[i:i+P, j:j+P, k:k+P]
                x = torch.tensor(patch, dtype=torch.float32
                                 ).unsqueeze(0).unsqueeze(0).to(device)
                corr = model(x).squeeze().cpu().numpy()   # (P,P,P)
                correction_sum[i:i+P, j:j+P, k:k+P] += corr * window
                weight_sum[i:i+P, j:j+P, k:k+P]     += window

    # Avoid division by zero at edges not covered by any patch
    mask = weight_sum > 0
    correction = np.zeros_like(obs)
    correction[mask] = correction_sum[mask] / weight_sum[mask]

    corrected = _norm01(np.clip(obs - correction, 0, None))

    if out_path:
        write_patt(corrected, out_path,
                   label=f"corrected_{label}")
        print(f"Wrote corrected map to {out_path}")

    return corrected


# ── Back-transform ─────────────────────────────────────────────────────────────

def back_transform_3d(corrected_map: np.ndarray,
                      spots_list,
                      normalise_to_obs: bool = True) -> Dict:
    """
    Extract corrected |F_hkl| values from a corrected Patterson map
    using the 3D inverse FFT relationship.

    The Patterson map P(u,v,w) = Σ_hkl |F_hkl|² exp(2πi(hu+kv+lw))
    so |F_hkl|² = FT[ P(u,v,w) ] evaluated at (h,k,l).

    Since P is real and centrosymmetric, we use rfftn and take the
    real part of the FFT coefficients.

    Parameters
    ----------
    corrected_map    : (N,N,N) corrected Patterson map
    spots_list       : SpotsList object (provides H,K,L and Fc for scaling)
    normalise_to_obs : if True, scale corrected Fo to match mean observed Fo

    Returns
    -------
    dict mapping (H,K,L) → Fo_corrected (float)
    """
    N = corrected_map.shape[0]

    # FFT of Patterson → |F|² at each grid point
    # Use real FFT: Patterson is real → F[h,k,l] is Hermitian
    # The real part of the FFT gives the cosine transform = Patterson coefficients
    F2 = np.real(np.fft.fftn(corrected_map))   # (N,N,N), values are |F_hkl|²

    corrections = {}
    for spot in spots_list.spots:
        h, k, l = spot.H, spot.K, spot.L

        # Map miller indices to FFT grid indices (with wrapping for negative h,k,l)
        ih = h % N
        ik = k % N
        il = l % N

        I_corr = F2[ih, ik, il]

        # I = F², F = sqrt(|I|)
        Fo_corr = np.sqrt(max(0.0, float(I_corr)))
        corrections[(h, k, l)] = Fo_corr

    # Scale corrected Fo to match the mean of the observed Fo
    if normalise_to_obs:
        obs_fos  = [s.Fo for s in spots_list.spots
                    if s.Fo is not None and s.Fo > 0]
        corr_fos = [corrections[k] for k in corrections if corrections[k] > 0]
        if obs_fos and corr_fos:
            scale = np.mean(obs_fos) / np.mean(corr_fos)
            corrections = {k: v * scale for k, v in corrections.items()}

    return corrections


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_correction(corrections:     Dict,
                        spots_list,
                        filename:        Optional[str] = None,
                        label:           str = '') -> Dict:
    """
    Evaluate the quality of the Patterson CNN correction.

    Computes:
      - R1 before correction: Σ|Fc - Fo| / ΣFc
      - R1 after  correction: Σ|Fc - Fo_corr| / ΣFc
      - Pearson correlation before/after
      - Scatter plots Fc vs Fo and Fc vs Fo_corrected

    Parameters
    ----------
    corrections : dict (H,K,L) → Fo_corrected from back_transform_3d
    spots_list  : SpotsList with Fc and Fo populated
    filename    : if given, save plot to this path
    label       : title prefix for the plot

    Returns
    -------
    dict with keys: R1_before, R1_after, corr_before, corr_after
    """
    Fc_vals, Fo_vals, Fo_corr_vals = [], [], []

    for spot in spots_list.spots:
        if spot.Fc is None or spot.Fo is None:
            continue
        if spot.Fo <= 0 or spot.Fc <= 0:
            continue
        key = (spot.H, spot.K, spot.L)
        if key not in corrections:
            continue
        Fo_c = corrections[key]
        if Fo_c <= 0:
            continue
        Fc_vals.append(spot.Fc)
        Fo_vals.append(spot.Fo)
        Fo_corr_vals.append(Fo_c)

    Fc   = np.array(Fc_vals)
    Fo   = np.array(Fo_vals)
    Fo_c = np.array(Fo_corr_vals)

    # Scale Fo to Fc scale for R1
    s_before = np.median(Fc / Fo)
    s_after  = np.median(Fc / Fo_c)

    R1_before = np.sum(np.abs(Fc - s_before * Fo))   / np.sum(Fc)
    R1_after  = np.sum(np.abs(Fc - s_after  * Fo_c)) / np.sum(Fc)

    corr_before = np.corrcoef(Fc, Fo)[0,1]
    corr_after  = np.corrcoef(Fc, Fo_c)[0,1]

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  N reflections: {len(Fc)}")
    print(f"  R1 before correction: {R1_before:.4f}")
    print(f"  R1 after  correction: {R1_after:.4f}")
    print(f"  Pearson r before: {corr_before:.4f}")
    print(f"  Pearson r after:  {corr_after:.4f}")
    print(f"{'='*50}\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    lim = max(Fc.max(), Fo.max(), Fo_c.max()) * 1.05

    for ax, fo, r, corr, title in zip(
        axes,
        [Fo, Fo_c],
        [R1_before, R1_after],
        [corr_before, corr_after],
        ['Fc vs Fo (observed)', 'Fc vs Fo (CNN corrected)']
    ):
        ax.scatter(Fc, fo * (np.median(Fc/fo)), s=3, alpha=0.4, c='C0')
        ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5)
        ax.set_xlabel('Fc')
        ax.set_ylabel('Fo (scaled)')
        ax.set_title(title)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.text(0.05, 0.92, f'R1 = {r:.4f}\nr = {corr:.4f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top')

    plt.suptitle(label, fontsize=11)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=200)
        print(f"Saved plot to {filename}")
    else:
        plt.show()
    plt.close()

    return {
        'R1_before':   R1_before,
        'R1_after':    R1_after,
        'corr_before': corr_before,
        'corr_after':  corr_after,
        'n':           len(Fc),
    }


# ── Convenience: full pipeline ────────────────────────────────────────────────

def run_training(obs_cal_pairs:  List[Tuple[str, str]],
                 patch_size:     int   = 16,
                 stride_train:   int   = 4,
                 base_ch:        int   = 32,
                 n_blocks:       int   = 8,
                 n_epochs:       int   = 100,
                 batch_size:     int   = 16,
                 lr:             float = 1e-3,
                 device:         str   = 'cuda',
                 checkpoint:     str   = 'patterson_filter.pt',
                 val_fraction:   float = 0.15) -> PattersonFilterCNN:
    """
    One-call training pipeline.

    Parameters
    ----------
    obs_cal_pairs : list of (obs.patt, cal.patt) file pairs
    patch_size    : size of 3D training patches
    stride_train  : patch extraction stride (smaller = more patches, more overlap)
    base_ch       : channels in residual blocks
    n_blocks      : number of residual blocks
                    receptive field = 1 + n_blocks*4 voxels
    n_epochs      : training epochs
    batch_size    : training batch size
    lr            : initial learning rate
    device        : 'cuda' or 'cpu'
    checkpoint    : path to save best model
    val_fraction  : fraction of datasets held out for validation

    Returns
    -------
    trained PattersonFilterCNN
    """
    print(f"Building patch dataset from {len(obs_cal_pairs)} file pairs...")
    full_ds = PattersonPatchDataset.from_patt_pairs(
        obs_cal_pairs, patch_size=patch_size, stride=stride_train
    )
    train_ds, val_ds = full_ds.split(val_fraction=val_fraction)

    model = PattersonFilterCNN(
        patch_size=patch_size, base_ch=base_ch, n_blocks=n_blocks
    )
    trainer = PattersonTrainer(model, device=device, lr=lr)
    trainer.train(train_ds, val_ds,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  checkpoint_path=checkpoint)
    trainer.plot_losses(filename=checkpoint.replace('.pt', '_losses.png'))

    return model


def run_inference(obs_path:     str,
                  model_path:   str,
                  spots_list,
                  patch_size:   int  = 16,
                  stride:       int  = 8,
                  device:       str  = 'cuda',
                  out_patt:     Optional[str] = None,
                  plot_path:    Optional[str] = None) -> Dict:
    """
    One-call inference + evaluation pipeline.

    Parameters
    ----------
    obs_path    : observed .patt file
    model_path  : saved model checkpoint (.pt file)
    spots_list  : SpotsList with Fc and Fo populated
    patch_size  : must match training patch size
    stride      : sliding window stride for inference
    device      : 'cuda' or 'cpu'
    out_patt    : if given, save corrected Patterson map here
    plot_path   : if given, save Fc vs Fo comparison plot here

    Returns
    -------
    dict with R1_before, R1_after, corr_before, corr_after, n
    """
    model = PattersonTrainer.load(model_path, device=device)

    print("Correcting Patterson map...")
    corrected = correct_volume(
        obs_path, model,
        patch_size=patch_size, stride=stride,
        device=device, out_path=out_patt
    )

    print("Back-transforming to reciprocal space...")
    corrections = back_transform_3d(corrected, spots_list)

    label = Path(obs_path).stem
    results = evaluate_correction(
        corrections, spots_list,
        filename=plot_path, label=label
    )

    # Write corrected Fo back onto spot objects
    n_written = 0
    for spot in spots_list.spots:
        key = (spot.H, spot.K, spot.L)
        if key in corrections:
            spot.Fo_corrected = corrections[key]
            n_written += 1
    print(f"Written corrected Fo to {n_written} spots")

    return results
