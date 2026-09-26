from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rt_utils import read_config, safe_name


def first_present(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def read_source(path: Path, config: dict) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    fields = config["input"]
    sequence_column = first_present(frame, fields["sequence_columns"])
    species_column = first_present(frame, fields["species_columns"])
    project_column = first_present(frame, fields["project_columns"])
    modification_column = first_present(frame, fields["modification_columns"])
    retention_column = first_present(frame, fields["retention_columns"])
    effective_rt_column = first_present(frame, fields["effective_rt_columns"])
    if None in (sequence_column, species_column, retention_column, effective_rt_column):
        return pd.DataFrame()
    sequence = frame[sequence_column].fillna("").astype(str).str.strip().str.upper()
    species = frame[species_column].fillna("unknown").astype(str).str.strip()
    modifications = (
        frame[modification_column].fillna("").astype(str).str.strip()
        if modification_column is not None
        else pd.Series("", index=frame.index, dtype=str)
    )
    retention = pd.to_numeric(frame[retention_column], errors="coerce")
    effective_rt = pd.to_numeric(frame[effective_rt_column], errors="coerce")
    project = (
        frame[project_column].fillna("unknown").astype(str).str.strip()
        if project_column is not None
        else pd.Series(path.parents[1].name, index=frame.index, dtype=str)
    )
    result = pd.DataFrame(
        {
            "sequence_key": sequence,
            "seq": sequence,
            "species": species,
            "modifications": modifications,
            "normalized_rt": 100.0 * retention / effective_rt,
            "project": project,
        }
    )
    return result[
        (result["sequence_key"] != "")
        & result["modifications"].eq("")
        & result["normalized_rt"].notna()
        & (effective_rt > 0)
    ]


def write_grouped_pools(data: pd.DataFrame, prepared_root: Path) -> list[dict]:
    records = []
    for split_type, column in (("by_first_char", "first_char"), ("by_species", "species")):
        output_directory = prepared_root / split_type
        output_directory.mkdir(parents=True, exist_ok=True)
        for group, subset in data.groupby(column, sort=True):
            output = subset[["sequence_key", "seq", "rt", "species", "project"]].copy()
            output.to_parquet(output_directory / f"{safe_name(group)}.parquet", index=False)
            records.append(
                {
                    "split_type": split_type,
                    "group": group,
                    "peptide_count": output["sequence_key"].nunique(),
                    "project_count": output["project"].str.split(";").explode().nunique(),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    input_root = Path(config["paths"]["input_root"])
    prepared_root = Path(config["paths"]["prepared_root"])
    prepared_root.mkdir(parents=True, exist_ok=True)
    files = sorted(input_root.glob(config["input"]["glob"]))
    frames = [read_source(path, config) for path in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No eligible peptide observations were found in the raw parquet files")
    observations = pd.concat(frames, ignore_index=True)
    observations["first_char"] = observations["sequence_key"].str[0]
    pooled = (
        observations.groupby(["sequence_key", "seq", "species", "first_char"], as_index=False)
        .agg(rt=("normalized_rt", "mean"), project=("project", lambda values: ";".join(sorted(set(values)))))
    )
    pooled.to_parquet(prepared_root / "all_peptide_species_records.parquet", index=False)
    records = write_grouped_pools(pooled, prepared_root)
    pd.DataFrame(records).sort_values(["split_type", "peptide_count"], ascending=[True, False]).to_csv(
        prepared_root / "input_pool_counts.csv", index=False
    )


if __name__ == "__main__":
    main()
