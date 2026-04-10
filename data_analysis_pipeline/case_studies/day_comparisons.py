import os
import numpy as np
import matplotlib.pyplot as plt

from data_analysis_pipeline.stats_analysis.data_loading import DataLoader



def _to_2d_array(x):
    """Normalize loader outputs to a plain 2D numpy array."""
    if isinstance(x, dict):
        if "cutouts" in x:
            x = x["cutouts"]
        elif "data" in x:
            x = x["data"]

    arr = np.asarray(x)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D field, got shape {arr.shape}")
    return arr


def downsample_mean(field, factor_y=12, factor_x=None):
    """
    Simple block-mean downsampling.
    Assumes field shape divisible by factor.
    """
    field = _to_2d_array(field)
    if factor_x is None:
        factor_x = factor_y

    H, W = field.shape
    field = field[: H - (H % factor_y), : W - (W % factor_x)]
    h_new = field.shape[0] // factor_y
    w_new = field.shape[1] // factor_x
    return field.reshape(h_new, factor_y, w_new, factor_x).mean(axis=(1, 3))


def _infer_downsample_factors(hr_shape, lr_shape):
    fy = hr_shape[0] // lr_shape[0]
    fx = hr_shape[1] // lr_shape[1]
    if fy <= 0 or fx <= 0:
        raise ValueError(f"Invalid downsample factors from HR {hr_shape} to LR {lr_shape}")
    if hr_shape[0] % lr_shape[0] != 0 or hr_shape[1] % lr_shape[1] != 0:
        raise ValueError(
            f"HR shape {hr_shape} is not evenly divisible by LR shape {lr_shape}. "
            "Use an interpolation-based downsampling method instead."
        )
    return fy, fx


def plot_day_comparison(
    era5_upsampled,
    danra_hr,
    variable,
    date_str="unknown",
    save_path=None,
    cfg=None,
):
    """
    Creates a 2x3 comparison figure:

    Top row:
        ERA5 upsampled (to DANRA grid)
        DANRA (HR)
        Difference (DANRA - ERA5 upsampled)

    Bottom row:
        DANRA coarsened and re-upsampled (illustrative coarse-scale DANRA)
        ERA5 on the DANRA grid
        Difference (coarsened DANRA - ERA5)

    NOTE:
    - No true ERA5 native required.
    - Bottom row is purely illustrative.
    """

    from sbgm.plotting_utils import imshow_variable
    from sbgm.variable_utils import get_cmap_for_variable

    era5_upsampled = _to_2d_array(era5_upsampled)
    danra_hr = _to_2d_array(danra_hr)

    if era5_upsampled.shape != danra_hr.shape:
        raise ValueError(
            f"ERA5 upsampled shape {era5_upsampled.shape} and DANRA HR shape {danra_hr.shape} must match"
        )

    # --- Create artificial coarse representation ---
    factor_y, factor_x = 12, 12
    danra_lr = downsample_mean(danra_hr, factor_y=factor_y, factor_x=factor_x)

    # Upsample back (nearest) for visual alignment
    danra_lr_up = np.repeat(np.repeat(danra_lr, factor_y, axis=0), factor_x, axis=1)

    diff_hr = danra_hr - era5_upsampled
    diff_lr = danra_lr_up - era5_upsampled

    vmin = min(np.nanmin(era5_upsampled), np.nanmin(danra_hr))
    vmax = max(np.nanmax(era5_upsampled), np.nanmax(danra_hr))
    diff_abs = max(np.nanmax(np.abs(diff_hr)), np.nanmax(np.abs(diff_lr)))

    cmap = get_cmap_for_variable(variable)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # --- TOP ROW (HR grid) ---
    im0 = imshow_variable(axes[0, 0], era5_upsampled, variable=variable, vmin=vmin, vmax=vmax, cmap=cmap, cfg=cfg)
    axes[0, 0].set_title("ERA5 (bilinear → DANRA grid)", fontsize=12)

    im1 = imshow_variable(axes[0, 1], danra_hr, variable=variable, vmin=vmin, vmax=vmax, cmap=cmap, cfg=cfg)
    axes[0, 1].set_title("DANRA (high-resolution truth)", fontsize=12)

    im2 = imshow_variable(axes[0, 2], diff_hr, variable=variable, vmin=-diff_abs, vmax=diff_abs, cmap="RdBu_r", cfg=cfg)
    axes[0, 2].set_title("Difference (DANRA − ERA5, HR grid)", fontsize=12)

    # --- BOTTOM ROW (illustrative coarse-scale comparison) ---
    im3 = imshow_variable(axes[1, 0], danra_lr_up, variable=variable, vmin=vmin, vmax=vmax, cmap=cmap, cfg=cfg)
    axes[1, 0].set_title("DANRA (coarsened → upsampled)", fontsize=12)

    im4 = imshow_variable(axes[1, 1], era5_upsampled, variable=variable, vmin=vmin, vmax=vmax, cmap=cmap, cfg=cfg)
    axes[1, 1].set_title("ERA5 (on DANRA grid)", fontsize=12)

    im5 = imshow_variable(axes[1, 2], diff_lr, variable=variable, vmin=-diff_abs, vmax=diff_abs, cmap="RdBu_r", cfg=cfg)
    axes[1, 2].set_title("Difference (coarse DANRA − ERA5)", fontsize=12)

    for ax in axes.flatten():
        ax.axis("off")

    cbar_main = fig.colorbar(im1, ax=[axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], fraction=0.025, pad=0.02)
    cbar_main.set_label(f"{variable} (shared scale)", fontsize=11)

    cbar_diff = fig.colorbar(im2, ax=[axes[0, 2], axes[1, 2]], fraction=0.025, pad=0.02)
    cbar_diff.set_label(f"{variable} difference", fontsize=11)

    fig.suptitle(
        f"{variable.upper()} comparison on {date_str}\n"
        "Top: High-resolution comparison (DANRA vs ERA5)\n"
        "Bottom: Coarse-scale comparison (DANRA aggregated vs ERA5)",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # fig.text(
    #     0.5,
    #     0.01,
    #     "Note: Coarsened DANRA is shown to illustrate that aggregating high-resolution data "
    #     "does not reproduce ERA5 structures, as the datasets originate from different physical models.",
    #     ha="center",
    #     fontsize=10,
    # )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def load_day_fields(
    base_dir,
    variable,
    date_str,
    hr_model="DANRA",
    lr_model="ERA5",
    hr_domain_size=(589, 789),
    lr_domain_size=(589, 789),
    hr_split="all",
    lr_split="all",
    hr_crop_region=None,
    lr_crop_region=None,
):
    """
    Load the two fields needed for the day-comparison plot.

    Returns
    -------
    era5_upsampled : np.ndarray
        ERA5 already regridded / upsampled to the DANRA grid.
    danra_hr : np.ndarray
        DANRA field on the HR grid.

    Notes
    -----
    Both fields are loaded through the same DataLoader logic
    used in the statistics pipeline.
    """
    hr_loader = DataLoader(
        base_dir=base_dir,
        n_workers=1,
        variable=variable,
        model=hr_model,
        domain_size=list(hr_domain_size),
        split=hr_split,
        crop_region=list(hr_crop_region) if hr_crop_region is not None else None,
        verbose=False,
    )

    lr_loader = DataLoader(
        base_dir=base_dir,
        n_workers=1,
        variable=variable,
        model=lr_model,
        domain_size=list(lr_domain_size),
        split=lr_split,
        crop_region=list(lr_crop_region) if lr_crop_region is not None else None,
        verbose=False,
    )

    danra_day = hr_loader.load_single_day(date_str)
    era5_day = lr_loader.load_single_day(date_str)

    danra_hr = _to_2d_array(danra_day)
    era5_upsampled = _to_2d_array(era5_day)

    return era5_upsampled, danra_hr


def make_day_comparison_from_loaders(
    base_dir,
    variable,
    date_str,
    save_path,
    hr_model="DANRA",
    lr_model="ERA5",
    hr_domain_size=(589, 789),
    lr_domain_size=(589, 789),
    hr_split="all",
    lr_split="all",
    hr_crop_region=None,
    lr_crop_region=None,
):
    era5_upsampled, danra_hr = load_day_fields(
        base_dir=base_dir,
        variable=variable,
        date_str=date_str,
        hr_model=hr_model,
        lr_model=lr_model,
        hr_domain_size=hr_domain_size,
        lr_domain_size=lr_domain_size,
        hr_split=hr_split,
        lr_split=lr_split,
        hr_crop_region=hr_crop_region,
        lr_crop_region=lr_crop_region,
    )

    plot_day_comparison(
        era5_upsampled=era5_upsampled,
        danra_hr=danra_hr,
        variable=variable,
        date_str=date_str,
        save_path=save_path,
        cfg=None,
    )



if __name__ == "__main__":

    base_dir = "/scratch/project_465002493/quistgaa/Data/Data_DiffMod"

    variable = "prcp"
    date_str = "20000222"

    save_path = f"./{variable}_{date_str}_comparison.png"

    # Optional crop (same as your stats pipeline)
    crop_region = None
    # Example:
    # crop_region = [170, 350, 340, 520]

    make_day_comparison_from_loaders(
        base_dir=base_dir,
        variable=variable,
        date_str=date_str,
        save_path=save_path,

        hr_model="DANRA",
        lr_model="ERA5",

        hr_domain_size=(589, 789),
        lr_domain_size=(589, 789),

        hr_split="all",
        lr_split="all",

        hr_crop_region=crop_region,
        lr_crop_region=crop_region,
    )