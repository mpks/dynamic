"""
patterson_cnn.py

CNN that learns to predict the deformation function between
observed and calculated Patterson maps.

Architecture:
    - Input:  observed Patterson map (N x N) + conditioning vector
    - Output: corrected Patterson map (N x N)
    - The network learns: Patterson_obs → Patterson_cal

The conditioning vector encodes dataset-level properties
(CV, dynamic range, mean I/sigma etc.) so the network knows
how strong the dynamical effects are.

Training pairs: (Patterson_obs, Patterson_cal) from known structures.
At inference: feed Patterson_obs from new dataset → get corrected map
→ back-transform to get corrected |F_hkl|.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from patterson import PattersonMap, compute_dataset_conditioning


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class PattersonDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of PattersonMap objects.

    Each sample is:
        x : (1, N, N) tensor  — normalised observed Patterson map
        c : (C,)     tensor  — conditioning vector
        y : (1, N, N) tensor  — normalised calculated Patterson map (target)
    """

    def __init__(self,
                 patterson_maps: List[PattersonMap],
                 conditioning_vectors: Optional[np.ndarray] = None,
                 augment: bool = True):
        """
        Parameters
        ----------
        patterson_maps       : list of PattersonMap objects
        conditioning_vectors : shape (n_datasets, C) — one row per dataset.
                               If None, a zero vector is used.
        augment              : if True, apply random flips/rotations
        """
        self.maps    = patterson_maps
        self.augment = augment

        # Build a lookup from dataset_id to conditioning vector
        self.cond_lookup = {}
        if conditioning_vectors is not None:
            # conditioning_vectors is a dict: {dataset_id: np.ndarray}
            self.cond_lookup = conditioning_vectors
        
        # Infer conditioning size
        sample_cond = next(iter(self.cond_lookup.values()), None)
        self.cond_size = len(sample_cond) if sample_cond is not None else 8

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        pm = self.maps[idx]

        # Maps — add channel dimension
        x = torch.tensor(pm.map_obs, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(pm.map_cal, dtype=torch.float32).unsqueeze(0)

        # Conditioning vector
        cond = self.cond_lookup.get(pm.dataset_id,
                                    np.zeros(self.cond_size, dtype=np.float32))
        c = torch.tensor(cond, dtype=torch.float32)

        # Augmentation — Patterson maps have inversion symmetry P(-r) = P(r)
        # so we can flip both axes independently
        if self.augment:
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            # 90-degree rotations also valid due to Patterson symmetry
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[1, 2])

        return x, c, y


# -----------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------

class ConditioningInjector(nn.Module):
    """
    Projects the conditioning vector to a spatial bias map
    that is added to feature maps at each scale.
    Allows the network to modulate its correction strength
    based on dataset properties.
    """
    def __init__(self, cond_size: int, n_channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_size, n_channels * 2),
            nn.ReLU(),
            nn.Linear(n_channels * 2, n_channels * 2)
        )
        self.n_channels = n_channels

    def forward(self, features: torch.Tensor,
                conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM (Feature-wise Linear Modulation):
            out = gamma * features + beta
        where gamma, beta are predicted from conditioning vector.
        """
        params = self.fc(conditioning)              # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)       # each (B, C)

        # Reshape for broadcasting over spatial dims
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)

        return gamma * features + beta


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU × 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock + MaxPool downsampling."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv  = ConvBlock(in_ch, out_ch)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Upsample + skip connection + ConvBlock."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch,
                                        kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd-sized inputs
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -----------------------------------------------------------------------
# Main model — conditioned U-Net
# -----------------------------------------------------------------------

class PattersonCNN(nn.Module):
    """
    Conditioned U-Net that maps observed Patterson maps to
    calculated Patterson maps.

    Architecture:
        Encoder: 4 levels of downsampling
        Bottleneck: deepest feature representation
        Decoder: 4 levels of upsampling with skip connections
        Conditioning: FiLM injection at each encoder level

    Input:
        x    : (B, 1, N, N) — observed Patterson map
        cond : (B, C)       — conditioning vector

    Output:
        (B, 1, N, N) — predicted calculated Patterson map
    """

    def __init__(self,
                 grid_size:   int = 64,
                 cond_size:   int = 8,
                 base_ch:     int = 32):
        """
        Parameters
        ----------
        grid_size : spatial size of Patterson maps (N)
        cond_size : length of conditioning vector
        base_ch   : base number of channels (doubles at each level)
        """
        super().__init__()

        ch = base_ch
        self.grid_size = grid_size
        self.cond_size = cond_size

        # Encoder
        self.enc1 = EncoderBlock(1,      ch)      # 64 → 32
        self.enc2 = EncoderBlock(ch,     ch*2)    # 32 → 16
        self.enc3 = EncoderBlock(ch*2,   ch*4)    # 16 → 8
        self.enc4 = EncoderBlock(ch*4,   ch*8)    # 8  → 4

        # Bottleneck
        self.bottleneck = ConvBlock(ch*8, ch*16)  # 4  → 4

        # Decoder
        self.dec4 = DecoderBlock(ch*16,  ch*8)   # 4  → 8
        self.dec3 = DecoderBlock(ch*8,   ch*4)   # 8  → 16
        self.dec2 = DecoderBlock(ch*4,   ch*2)   # 16 → 32
        self.dec1 = DecoderBlock(ch*2,   ch)     # 32 → 64

        # Output — predict residual (obs + residual = cal)
        self.output_conv = nn.Conv2d(ch, 1, kernel_size=1)

        # Conditioning injectors — one per encoder level
        self.cond1 = ConditioningInjector(cond_size, ch)
        self.cond2 = ConditioningInjector(cond_size, ch*2)
        self.cond3 = ConditioningInjector(cond_size, ch*4)
        self.cond4 = ConditioningInjector(cond_size, ch*8)

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:

        # Encoder path — save skips
        x1, skip1 = self.enc1(x)
        x1 = self.cond1(x1, cond)

        x2, skip2 = self.enc2(x1)
        x2 = self.cond2(x2, cond)

        x3, skip3 = self.enc3(x2)
        x3 = self.cond3(x3, cond)

        x4, skip4 = self.enc4(x3)
        x4 = self.cond4(x4, cond)

        # Bottleneck
        b = self.bottleneck(x4)

        # Decoder path
        d = self.dec4(b,  skip4)
        d = self.dec3(d,  skip3)
        d = self.dec2(d,  skip2)
        d = self.dec1(d,  skip1)

        # Predict residual and add to input (skip connection at map level)
        residual = self.output_conv(d)
        return x + residual


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

class PattersonTrainer:
    """
    Handles training, validation, and checkpointing of PattersonCNN.
    """

    def __init__(self,
                 model:       PattersonCNN,
                 device:      str = 'cuda',
                 lr:          float = 1e-3,
                 weight_decay: float = 1e-4):

        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.Adam(model.parameters(),
                                       lr=lr,
                                       weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=5, factor=0.5, verbose=True
        )
        self.train_losses = []
        self.val_losses   = []

    def loss_fn(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Combined L1 + MSE loss.
        L1 is robust to outliers (important for Patterson maps
        which can have sharp origin peaks).
        MSE penalises large deviations.
        """
        l1  = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        return 0.5 * l1 + 0.5 * mse

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, c, y in loader:
            x = x.to(self.device)
            c = c.to(self.device)
            y = y.to(self.device)

            self.opt.zero_grad()
            pred = self.model(x, c)
            loss = self.loss_fn(pred, y)
            loss.backward()

            # Gradient clipping — Patterson maps can have large dynamic range
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for x, c, y in loader:
            x = x.to(self.device)
            c = c.to(self.device)
            y = y.to(self.device)
            pred = self.model(x, c)
            total_loss += self.loss_fn(pred, y).item()
        return total_loss / len(loader)

    def train(self,
              train_loader: DataLoader,
              val_loader:   DataLoader,
              n_epochs:     int = 100,
              checkpoint_path: str = 'patterson_cnn.pt'):

        best_val = float('inf')

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.val_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch:04d} | "
                  f"train loss: {train_loss:.6f} | "
                  f"val loss:   {val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'epoch':       epoch,
                    'model_state': self.model.state_dict(),
                    'opt_state':   self.opt.state_dict(),
                    'val_loss':    val_loss,
                    'grid_size':   self.model.grid_size,
                    'cond_size':   self.model.cond_size,
                }, checkpoint_path)
                print(f"  → Saved best model (val loss {best_val:.6f})")

    def plot_losses(self, filename: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.train_losses, label='Train')
        ax.plot(self.val_losses,   label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Patterson CNN Training')
        ax.legend()
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=200)
        else:
            plt.show()


# -----------------------------------------------------------------------
# Inference — correct a new dataset
# -----------------------------------------------------------------------

@torch.no_grad()
def correct_patterson_map(model:         PattersonCNN,
                          map_obs:       np.ndarray,
                          conditioning:  np.ndarray,
                          device:        str = 'cuda') -> np.ndarray:
    """
    Run inference on a single observed Patterson map.

    Parameters
    ----------
    model        : trained PattersonCNN
    map_obs      : (N, N) observed Patterson map (normalised)
    conditioning : (C,) conditioning vector for this dataset
    device       : 'cuda' or 'cpu'

    Returns
    -------
    map_corrected : (N, N) corrected Patterson map
    """
    model.eval()
    model.to(device)

    x = torch.tensor(map_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    c = torch.tensor(conditioning, dtype=torch.float32).unsqueeze(0)

    x = x.to(device)
    c = c.to(device)

    pred = model(x, c)
    return pred.squeeze().cpu().numpy()


def back_transform_patterson(map_corrected: np.ndarray,
                              spots,
                              grid_size: int = 64) -> dict:
    """
    Extract corrected |F_hkl| values from a corrected Patterson map
    by projecting back onto the measured reflection positions.

    This uses the inverse relationship:
        I_hkl_corrected ∝ integral of P_corrected * cos(2pi*(h*u + k*v)) du dv

    Which is just the discrete Fourier coefficient at (h, k).

    Parameters
    ----------
    map_corrected : (N, N) corrected Patterson map
    spots         : list of Spot objects (same image)
    grid_size     : must match the map grid size

    Returns
    -------
    dict mapping (H, K, L) → corrected Fo value
    """
    N = grid_size
    u = np.linspace(0, 1, N, endpoint=False)
    v = np.linspace(0, 1, N, endpoint=False)
    uu, vv = np.meshgrid(u, v)
    du_dv = (1.0 / N) ** 2   # area element

    corrections = {}
    for spot in spots:
        h = spot.H
        k = spot.K

        # Project Patterson onto this (h,k) — Fourier coefficient
        phase  = 2.0 * np.pi * (h * uu + k * vv)
        I_corr = np.sum(map_corrected * np.cos(phase)) * du_dv

        # I = F^2, so F = sqrt(|I|)
        Fo_corrected = np.sqrt(np.abs(I_corr))
        corrections[(spot.H, spot.K, spot.L)] = Fo_corrected

    return corrections


def apply_patterson_correction(model:        PattersonCNN,
                               spots_list,
                               device:       str = 'cuda',
                               grid_size:    int = 64,
                               verbose:      bool = True):
    """
    Full pipeline: given a SpotsList, compute per-image Patterson maps,
    run CNN correction, back-transform, and write corrected Fo values
    back onto each Spot object.

    Parameters
    ----------
    model      : trained PattersonCNN
    spots_list : SpotsList object to correct
    device     : 'cuda' or 'cpu'
    grid_size  : Patterson map grid size
    verbose    : print progress
    """
    from patterson import (compute_patterson_2d,
                           compute_dataset_conditioning)

    # Dataset-level conditioning
    conditioning = compute_dataset_conditioning(spots_list)

    groups = spots_list.group_by_image()
    n_corrected = 0

    for image_id, image_spots in groups.items():

        if len(image_spots) < 3:
            continue

        # Observed Patterson for this image
        map_obs, _ = compute_patterson_2d(
            image_spots.spots,
            grid_size=grid_size,
            normalise=True
        )

        # CNN correction
        map_corrected = correct_patterson_map(
            model, map_obs, conditioning, device=device
        )

        # Back-transform to get corrected Fo per reflection
        corrections = back_transform_patterson(
            map_corrected, image_spots.spots, grid_size=grid_size
        )

        # Write corrected values back to spot objects
        for spot in image_spots.spots:
            key = (spot.H, spot.K, spot.L)
            if key in corrections:
                spot.Fo_corrected = corrections[key]
                n_corrected += 1

    if verbose:
        print(f"Corrected {n_corrected} spots in dataset "
              f"{spots_list.output_prefix}")


# -----------------------------------------------------------------------
# Convenience: build datasets and loaders from SpotsList objects
# -----------------------------------------------------------------------

def build_loaders(spots_lists: list,
                  grid_size:      int   = 64,
                  val_fraction:   float = 0.15,
                  batch_size:     int   = 32,
                  num_workers:    int   = 4,
                  augment:        bool  = True):
    """
    Build train/val DataLoaders from a list of SpotsList objects.

    Parameters
    ----------
    spots_lists    : list of SpotsList objects
    grid_size      : Patterson map grid size
    val_fraction   : fraction of datasets held out for validation
    batch_size     : training batch size
    num_workers    : DataLoader workers
    augment        : use data augmentation during training

    Returns
    -------
    train_loader, val_loader, model_kwargs
        model_kwargs contains grid_size and cond_size for PattersonCNN init
    """
    from patterson import (compute_patterson_maps_for_dataset,
                           compute_dataset_conditioning)
    from torch.utils.data import random_split

    # Split datasets into train/val at the dataset level
    # (not at the image level, to avoid data leakage)
    n_val   = max(1, int(len(spots_lists) * val_fraction))
    n_train = len(spots_lists) - n_val

    idx      = np.random.permutation(len(spots_lists))
    train_sl = [spots_lists[i] for i in idx[:n_train]]
    val_sl   = [spots_lists[i] for i in idx[n_train:]]

    def build_maps_and_conds(sl_list):
        all_maps  = []
        cond_dict = {}
        for sl in sl_list:
            maps = compute_patterson_maps_for_dataset(
                sl, grid_size=grid_size, verbose=False
            )
            cond = compute_dataset_conditioning(sl)
            cond_dict[sl.output_prefix] = cond
            all_maps.extend(maps)
        return all_maps, cond_dict

    print(f"Building training maps ({n_train} datasets)...")
    train_maps, train_conds = build_maps_and_conds(train_sl)

    print(f"Building validation maps ({n_val} datasets)...")
    val_maps, val_conds = build_maps_and_conds(val_sl)

    print(f"Train: {len(train_maps)} images | Val: {len(val_maps)} images")

    # Infer conditioning size
    cond_size = len(next(iter(train_conds.values())))

    train_ds = PattersonDataset(train_maps, train_conds, augment=augment)
    val_ds   = PattersonDataset(val_maps,   val_conds,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    model_kwargs = dict(grid_size=grid_size, cond_size=cond_size)

    return train_loader, val_loader, model_kwargs
