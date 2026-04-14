"""
example_patterson.py

Example showing how to use patterson.py and patterson_cnn.py
with your SpotsList objects.
"""

import torch
import numpy as np
from patterson import (
    compute_patterson_maps_for_dataset,
    compute_patterson_maps_for_datasets,
    compute_dataset_conditioning,
    plot_patterson_pair
)
from patterson_cnn import (
    PattersonCNN,
    PattersonTrainer,
    build_loaders,
    apply_patterson_correction
)


# -----------------------------------------------------------------------
# 1. Inspect a single dataset — visualise Patterson maps
# -----------------------------------------------------------------------

def inspect_dataset(spots_list, grid_size=64):
    """Quick look at Patterson maps for one dataset."""

    maps = compute_patterson_maps_for_dataset(
        spots_list, grid_size=grid_size, verbose=True
    )

    print(f"\nComputed {len(maps)} Patterson maps")
    print(f"Conditioning vector: {compute_dataset_conditioning(spots_list)}")

    # Plot first 3 images
    for pm in maps[:3]:
        plot_patterson_pair(pm,
                            filename=f"{spots_list.output_prefix}_"
                                     f"patterson_{pm.image_id:04d}.png")


# -----------------------------------------------------------------------
# 2. Train the CNN on a list of datasets
# -----------------------------------------------------------------------

def train(spots_lists,
          grid_size=64,
          n_epochs=100,
          batch_size=32,
          device='cuda',
          checkpoint='patterson_cnn_best.pt'):

    # Build DataLoaders — splits at dataset level to avoid leakage
    train_loader, val_loader, model_kwargs = build_loaders(
        spots_lists,
        grid_size=grid_size,
        val_fraction=0.15,
        batch_size=batch_size,
        augment=True
    )

    # Initialise model
    model   = PattersonCNN(**model_kwargs, base_ch=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    trainer = PattersonTrainer(model, device=device, lr=1e-3)
    trainer.train(train_loader, val_loader,
                  n_epochs=n_epochs,
                  checkpoint_path=checkpoint)
    trainer.plot_losses()

    return model


# -----------------------------------------------------------------------
# 3. Load a trained model and apply correction to a new dataset
# -----------------------------------------------------------------------

def load_and_apply(spots_list,
                   checkpoint='patterson_cnn_best.pt',
                   grid_size=64,
                   device='cuda'):

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device)
    model = PattersonCNN(
        grid_size=ckpt['grid_size'],
        cond_size=ckpt['cond_size'],
        base_ch=32
    )
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded model from epoch {ckpt['epoch']}, "
          f"val loss {ckpt['val_loss']:.6f}")

    # Apply correction — writes Fo_corrected onto each Spot
    apply_patterson_correction(model, spots_list,
                               device=device,
                               grid_size=grid_size,
                               verbose=True)

    # Visualise result
    spots_list.plot_fc_vs_predicted()


# -----------------------------------------------------------------------
# Example entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':

    # Load your datasets (replace with your actual loading code)
    # from dynamic.spots import SpotsList
    # spots_lists = [SpotsList.from_npz(f) for f in npz_files]

    # --- Quick inspection of one dataset ---
    # inspect_dataset(spots_lists[0])

    # --- Train on all datasets ---
    # model = train(spots_lists, grid_size=64, n_epochs=100, device='cuda')

    # --- Apply to a new dataset ---
    # load_and_apply(new_spots_list, checkpoint='patterson_cnn_best.pt')

    print("See function docstrings for usage.")
