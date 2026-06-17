import matplotlib.pyplot as plt
import numpy as np


def plot_detector_image(image, positions, miller_indices,
                        intensities, npx, npy, wavelength_A, distance_mm,
                        pixel_size_mm, beam_centre_px, out_path,
                        top_spots=20):
    """
    Save a raster PNG of the detector image as it would appear
    in dials.image_viewer — binary-style greyscale with red
    Miller index labels for the top_spots brightest spots.

    The raster is exactly npx x npy pixels (one pixel per
    detector pixel).  Matplotlib renders it at higher DPI so
    the labels are readable, but the pixel grid matches the
    detector 1:1.
    """
    k0 = 1.0 / wavelength_A
    cx, cy = beam_centre_px

    # Project spot positions to pixel coordinates
    px_coords = []
    for pos in positions:
        kx, ky, kz_lattice = pos[0], pos[1], pos[2]
        kz_beam = k0 + kz_lattice
        if kz_beam <= 0:
            px_coords.append(None)
            continue
        dx = (kx / kz_beam) * distance_mm
        dy = -(ky / kz_beam) * distance_mm
        px_x = cx + dx / pixel_size_mm
        px_y = cy + dy / pixel_size_mm
        px_coords.append((px_x, px_y))

    # Select top spots by intensity
    i_sorted = np.argsort(intensities)[::-1]
    top_idx = set(i_sorted[:top_spots].tolist())

    # Figure sized so that 1 data unit = 1 detector pixel.
    # Use dpi=100 and figure size in inches = npx/100 x npy/100
    # so the pixel grid is exact.
    dpi = 100
    fig_w = npx / dpi
    fig_h = npy / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Display the detector image — binary greyscale like
    # dials.image_viewer (log scale to show weak spots)
    display = np.log1p(image.astype(np.float32))
    ax.imshow(
        display,
        cmap='binary',
        origin='upper',
        extent=[0, npx, npy, 0],
        interpolation='nearest',
        aspect='equal',
    )

    # Overlay labelled spots
    for i, (pos, miller, ints) in enumerate(
        zip(px_coords, miller_indices, intensities)
    ):
        if pos is None:
            continue
        px_x, px_y = pos
        if not (0 <= px_x < npx and 0 <= px_y < npy):
            continue

        if i in top_idx:
            h = int(miller[0])
            k = int(miller[1])
            l = int(miller[2])     # noqa: E741
            ax.plot(px_x, px_y, 'o',
                    color='red',
                    markersize=4,
                    markerfacecolor='none',
                    markeredgewidth=1)
            ax.text(
                px_x + 3, px_y - 3,
                f"{h},{k},{l}",
                color='red',
                fontsize=4,
                va='bottom',
                ha='left',
            )
        else:
            ax.plot(px_x, px_y, '.',
                    color='red',
                    markersize=1,
                    alpha=0.5)

    ax.set_xlim(0, npx)
    ax.set_ylim(npy, 0)
    ax.axis('off')

    plt.savefig(
        out_path,
        dpi=dpi * 2,   # 2x for readable labels
        bbox_inches='tight',
        pad_inches=0,
        facecolor='black',
    )
    plt.close(fig)
    print(f"Detector image plot saved: {out_path}")


def plot_spots(all_pos, all_miller, intensities,
               out_str, top_spots=60, out_path=None):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.15, 0.12, 0.80, 0.84])
    ax.set_xlabel(r'$k_{\rm x}\ (\rm \AA)$')
    ax.set_ylabel(r'$k_{\rm y}\ (\rm \AA)$')
    if len(intensities) > top_spots:
        icutoff = np.argsort(intensities)[-top_spots]
        intensity_cutoff = intensities[icutoff]
    else:
        intensity_cutoff = 0
    kxs, kys, ints_all = [], [], []
    kxs_other, kys_other = [], []
    for pos, miller, ints in zip(all_pos, all_miller, intensities):
        kx, ky, kz = pos
        if ints > intensity_cutoff:
            kxs.append(kx)
            kys.append(ky)
            ints_all.append(ints)
            label = f"{miller[0]} {miller[1]} {miller[2]}"
            plt.text(kx, ky + 0.05, label, color='blue', va='top',
                     ha='center', fontsize=4,
                     bbox=dict(facecolor='white', alpha=0.0,
                               edgecolor='none',
                               boxstyle='square, pad=0.3'))
        else:
            kxs_other.append(kx)
            kys_other.append(ky)
    ax.scatter(kxs_other, kys_other, marker='o', s=0.5,
               c='#BEBEBE', linewidths=0)
    ax.scatter(kxs, kys, marker='o',
               s=1 + 2 * abs(np.log(ints_all)),
               c='C3', linewidths=0, cmap='jet')
    out_file = f'image_{out_str}.png'
    if out_path is not None:
        out_file = out_path + '/' + out_file
    plt.savefig(out_file, dpi=400)
    plt.close(fig)


def plot_detector_labels(
    positions,
    miller_indices,
    intensities,
    npx,
    npy,
    wavelength_A,
    distance_mm,
    pixel_size_mm,
    beam_centre_px,
    out_path,
    top_spots=80,
):
    """
    Save a PNG showing spot positions on the detector with
    abTEM Miller index labels.  Use this to compare abTEM
    indexing against DIALS indexing of the same image.

    Only the top_spots brightest spots are labelled to keep
    the plot readable.
    """
    k0 = 1.0 / wavelength_A
    cx, cy = beam_centre_px

    # Project all spots to pixel coordinates
    px_coords = []
    for pos in positions:
        kx, ky, kz_lattice = pos[0], pos[1], pos[2]
        kz_beam = k0 + kz_lattice
        if kz_beam <= 0:
            px_coords.append(None)
            continue
        dx = (kx / kz_beam) * distance_mm
        dy = -(ky / kz_beam) * distance_mm
        px_x = cx + dx / pixel_size_mm
        px_y = cy + dy / pixel_size_mm
        px_coords.append((px_x, px_y))

    # Select top spots by intensity
    i_sorted = np.argsort(intensities)[::-1]
    top_idx = set(i_sorted[:top_spots].tolist())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, npx)
    ax.set_ylim(npy, 0)   # detector y increases downward
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("abTEM spot positions and Miller indices")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    for i, (pos, miller, ints) in enumerate(
        zip(px_coords, miller_indices, intensities)
    ):
        if pos is None:
            continue
        px_x, px_y = pos
        if not (0 <= px_x < npx and 0 <= px_y < npy):
            continue

        if i in top_idx:
            h = int(miller[0])
            k = int(miller[1])
            l = int(miller[2])     # noqa: E741
            size = 3 + 6 * (ints / intensities.max())
            ax.plot(px_x, px_y, 'o',
                    color='red', markersize=size)
            ax.text(
                px_x + 4, px_y - 4,
                f"{h},{k},{l}",
                color='yellow', fontsize=5,
                va='bottom', ha='left',
            )
        else:
            ax.plot(px_x, px_y, '.',
                    color='#666666', markersize=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Detector label plot saved: {out_path}")
