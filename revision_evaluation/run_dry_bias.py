"""
Run Group 3 dry-bias decomposition on saved precipitation fields
"""
import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path

import yaml

from sbgm.runtime import external_output

from .common_io import (
    RevisionInputs,
    resolve_inputs,
    write_table,
    write_run_metadata,
)
from .dry_bias_decomposition import (
    QUANTILES,
    add_counts,
    empty_counts,
    field_components,
    summarize_components,
    summarize_wet_values,
)
from .thresholds import SEASONS, WET_THRESHOLD, season_of


logger = logging.getLogger(__name__)


def run(config):
    config = dict(config)

    # Reuse date population frozen in Group 1
    frozen = Path(
        config["dates_file"] or Path(config["output_root"]) / "deterministic/dates.txt"
    )

    if not frozen.is_file():
        raise FileNotFoundError("Reuse Group 1 dates.txt via --dates-file or <output_root>/deterministic/dates.txt")

    config["dates_file"] = str(frozen.resolve())

    directory = external_output(Path(config["output_root"]) / "dry_bias")
    directory.mkdir(parents=True, exist_ok=False)

    settings = dict(
        stage="dry_bias",
        output=str(directory),
        wet_threshold=WET_THRESHOLD,
        conditional_quantiles=list(QUANTILES),
        conditioning=("Each method on its own precipitation >= wet threshold"),
        aggregation=("Pooled common land pixel-days within ALL or individual seasons"),
    )

    (directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            dict(config, dry_bias=settings),
            sort_keys=False,
        )
    )

    manifest = directory / "manifest.json"

    write_run_metadata(
        manifest,
        config, 
        status="running",
        **settings,
    )

    try: 
        reader = RevisionInputs(config)
        methods = ["danra", *config["methods"]]

        (directory / "dates.txt").write_text("\n".join(reader.dates) + "\n")
        (directory / "date_inventory.json").write_text(json.dumps(reader.inventory, indent=2) + "\n")

        # One accumulator for every method x season combo
        counts = {
            (method, season): empty_counts()
            for method in methods
            for season in SEASONS
        }

        # Only wet values need to be retained as quantiles cannot be reconstructed from dry values alone
        wet_chunks = defaultdict(list)

        n_valid_dates = 0

        for index, date in enumerate(reader.dates, 1):
            sample = reader.load_date(date)
            season = season_of(date)
            valid = sample["valid"]

            if valid.any():
                n_valid_dates += 1

            fields = {"danra": sample["observation"],
                      **sample["fields"],
                      }

            for method, field in fields.items():
                day_counts, wet_values = field_components(
                    field, valid, threshold=WET_THRESHOLD
                )

                # Each date contributes both to ALL and its own season
                for group in ("ALL", season):
                    add_counts(counts[(method, group)], day_counts)

                    if wet_values.size:
                        wet_chunks[(method, group)].append(wet_values)

            if (index == 1 or index % 25 == 0 or index == len(reader.dates)):
                logger.info("Processed %d/%d dates (%s); valid land pixels: %d", index, len(reader.dates), date, int(valid.sum()),)

        reader.check_unchanged()

        seasonal_rows = []
        conditional_rows = []

        for method in methods:
            # Full and seasonal occurrence/intensity decomposition
            for season in SEASONS:
                summary = summarize_components(counts[(method, season)])
                seasonal_rows.append(
                    dict(
                        method=method,
                        season=season,
                        subset="all_land",
                        wet_threshold=WET_THRESHOLD,
                        **summary,
                    )
                )

            # Main pooled conditional intensity distribution
            wet_stats = summarize_wet_values(wet_chunks[(method, "ALL")])

            conditional_rows.append(
                dict(
                    method=method,
                    season="ALL",
                    subset="all_land",
                    wet_threshold=WET_THRESHOLD,
                    **wet_stats,
                )
            )

        write_table(directory / "conditional_intensity.csv", conditional_rows)
        write_table(directory / "seasonal_decomposition.csv", seasonal_rows)
        write_table(directory / "input_files.csv", list(reader.files.values()))

        total = counts[("danra", "ALL")]["n_pixel_days"]

        write_run_metadata(
            manifest,
            config,
            status="complete",
            dates=reader.dates,
            n_dates=len(reader.dates),
            n_valid_dates=n_valid_dates,
            n_pixel_days=total,
            grid_shape=list(reader.shape),
            **settings,
        )

    except Exception as e:
        write_run_metadata(
            manifest,
            config,
            status="failed",
            error=str(e),
            **settings,
        )
        raise 

    logger.info("Completed dry-bias evaluation: %s", directory)

    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("atmo.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dates-file", type=Path)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = resolve_inputs(args.config, args.output_root, args.dates_file)

    print(run(config))


if __name__ == "__main__":
    main()