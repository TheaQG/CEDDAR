from pathlib import Path
import argparse
import csv

import numpy as np

from revision_manuscript_plots.data import legacy
from revision_evaluation.probabilistic_calibration import (
    daily_spread_skill,
)


def season(date):
    month = int(date[4:6])

    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def load_npz(path):
    with np.load(path) as data:
        return {
            key: np.asarray(data[key])
            for key in data.files
        }


def load_ensemble(path):
    data = load_npz(path)

    ens = np.asarray(data["ens"], dtype=float,)

    # [M,1,H,W] -> [M,H,W]
    if ens.ndim == 4 and ens.shape[1] == 1:
        ens = ens[:, 0]

    if ens.ndim != 3:
        raise ValueError(f"{path}: expected [M,H,W], got {ens.shape}")

    return ens


def load_hr(path):
    data = load_npz(path)

    for key in ("hr", "hr_phys", "danra", "HR",):
        if key in data:
            return np.asarray(data[key], dtype=float,).squeeze()

    raise KeyError(f"{path}: HR not found; keys={list(data)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--generation-dir", type=Path, required=True,)
    parser.add_argument("--output", type=Path, required=True,)

    args = parser.parse_args()

    root = args.generation_dir.expanduser().resolve()

    ens_dir = root / "ensembles_phys"
    lrhr_dir = root / "lr_hr_phys"

    # Same static Danish land mask used elsewhere.
    land = legacy.load_land_mask(root)
    land = np.asarray(land, dtype=bool,)

    ens_dates = {path.stem for path in ens_dir.glob("*.npz")}
    hr_dates = {path.stem for path in lrhr_dir.glob("*.npz")}

    dates = sorted(ens_dates & hr_dates)

    if not dates:
        raise RuntimeError("No common ensemble / HR dates found")

    print(f"Dates: {len(dates)}")

    rows = []

    for i, date in enumerate(dates):
        ensemble = load_ensemble(ens_dir / f"{date}.npz")
        observation = load_hr(lrhr_dir / f"{date}.npz")

        if observation.shape != land.shape:
            raise ValueError(f"{date}: observation shape {observation.shape} != mask {land.shape}")

        if ensemble.shape[1:] != land.shape:
            raise ValueError(f"{date}: ensemble shape {ensemble.shape} incompatible with mask {land.shape}")

        finite = (land & np.isfinite(observation) & np.all(np.isfinite(ensemble), axis=0,))

        # ------------------------------------------------------
        # All land
        # ------------------------------------------------------

        ens_all = ensemble[:, finite,]
        obs_all = observation[finite]

        metrics = daily_spread_skill(ens_all, obs_all,)

        rows.append({
                "date": date,
                "season": season(date),
                "subset": "all_land",
                **metrics,
            })

        # ------------------------------------------------------
        # Observed-wet land
        # ------------------------------------------------------

        wet = (finite & (observation >= 1.0))

        ens_wet = ensemble[:, wet,]
        obs_wet = observation[wet]

        metrics = daily_spread_skill(ens_wet, obs_wet,)

        rows.append({
                "date": date,
                "season": season(date),
                "subset": "observed_wet",
                **metrics,
            })

        if ((i + 1) % 50 == 0 or i == 0 or i + 1 == len(dates)):
            print(f"Processed {i + 1}/{len(dates)}")

    output = args.output.expanduser().resolve()

    output.parent.mkdir(parents=True, exist_ok=True,)

    with output.open("w", newline="",) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("date", "season", "subset", "mean_spread", "rmse", "n_cases",
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()