#!/usr/bin/env python3
"""Build the Phase-0 clean-room data artifacts. This script never trains a model."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from alphabase.peptide.mobility import mobility_to_ccs_bruker


ROOT = Path(
    os.environ.get("CCS_SCALING_ROOT", Path.cwd() / "ccs_scaling_runs")
).resolve()
PHASE0D = ROOT / "phase0d"
SOURCE = PHASE0D / "source_data"
ARTIFACTS = PHASE0D / "artifacts"
PXD_REPORT = Path(
    os.environ.get(
        "CCS_PXD019086_REPORT",
        SOURCE / "20201028_ExperimentalLibrary-Report.csv",
    )
).resolve()
RAW_ARCHIVE_ROOT = Path(
    os.environ.get("CCS_PXD019086_ARCHIVE_ROOT", SOURCE / "PXD019086_archives")
).resolve()
EXPERIMENT_ID = "full_data_clean_v1"

FIG1 = SOURCE / "SourceData_Figure_1.csv"
FIG4 = SOURCE / "SourceData_Figure_4.csv"
ALIGNMENT_NOTEBOOK = SOURCE / "CCS_Alignment.ipynb"

# These are independent outputs printed by the official CCS_Alignment notebook.
OFFICIAL_PIPELINE_COUNTS = {
    "all_filtered_observations_endogenous_plus_proteometools": 2_786_717,
    "filtered_endogenous_observations": 2_465_262,
    "filtered_proteometools_observations": 321_455,
    "per_run_abundance_collapsed_endogenous_plus_proteometools": 718_917,
    "final_endogenous_unique_modified_sequence_charge": 559_979,
    "aligned_proteometools_unique_modified_sequence_charge": 213_852,
    "endogenous_proteometools_precursor_overlap": 54_914,
    "paper_independent_proteometools_test": 155_004,
    "figure4_source_data_rows": 154_235,
}

RAW_ARCHIVE_EVIDENCE_ROWS = {
    "Results_Celegans.zip": 141_729,
    "Results_Drosophila.zip": 426_329,
    "Results_Ecoli.zip": 317_797,
    "Results_HeLa_LysC_LysN.zip": 209_472,
    "Results_HeLa_trypsin.zip": 352_110,
    "Results_Yeast.zip": 435_339,
    "Results_ProteomeTools_MissingGenes.zip": 67_457,
    "Results_ProteomeTools_Proteotypic.zip": 161_172,
    "Results_ProteomeTools_SRMatlas.zip": 102_067,
}

TRAINING_PROVENANCE = [
    "Meier_2021_Nature_Communications_SourceData_Figure_1",
    "ProteomeXchange:PXD019086",
    "ProteomeXchange:PXD010012",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_clean_training_provenance(items: list[str]) -> None:
    contaminated = [x for x in items if "pxd017703" in str(x).lower() or "diapasef" in str(x).lower()]
    assert not contaminated, f"PXD017703 contamination in clean training provenance: {contaminated}"


def canonical_mods(sequence: str, modified: str) -> tuple[str, str]:
    """Convert the two notations in official/Spectronaut files to AlphaBase mods/sites."""
    sequence = str(sequence).strip("_")
    modified = str(modified).strip("_")
    entries: set[tuple[int, str]] = {(i, "Carbamidomethyl@C") for i, aa in enumerate(sequence, 1) if aa == "C"}
    if modified.startswith("(ac)") or "[Acetyl (Protein N-term)]" in modified:
        entries.add((0, "Acetyl@Protein_N-term"))

    # Official MaxQuant shorthand and Spectronaut long form both place oxidation after M.
    plain_pos = 0
    i = 0
    while i < len(modified):
        ch = modified[i]
        if ch == "(":
            i = modified.find(")", i) + 1
            continue
        if ch == "[":
            i = modified.find("]", i) + 1
            continue
        if ch.isalpha() and ch.isupper():
            plain_pos += 1
            if ch == "M":
                tail = modified[i + 1 :]
                if tail.startswith("(ox)") or tail.startswith("[Oxidation (M)]"):
                    entries.add((plain_pos, "Oxidation@M"))
        i += 1
    ordered = sorted(entries, key=lambda x: (x[0], x[1]))
    return ";".join(x[1] for x in ordered), ";".join(str(x[0]) for x in ordered)


def add_canonical_fields(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [canonical_mods(s, m) for s, m in zip(df["sequence"], df["modified_sequence"])]
    df = df.copy()
    df["mods"] = [x[0] for x in pairs]
    df["mod_sites"] = [x[1] for x in pairs]
    df["nAA"] = df["sequence"].str.len().astype("int16")
    df["charge"] = df["charge"].astype("int8")
    df["precursor_key"] = (
        df["sequence"].astype(str) + "|" + df["charge"].astype(str) + "|" + df["mods"] + "|" + df["mod_sites"]
    )
    return df


def build_meier() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(FIG1)
    renamed = raw.rename(
        columns={
            "Modified sequence": "modified_sequence",
            "Sequence": "sequence",
            "Charge": "charge",
            "Mass": "mass",
            "m/z": "precursor_mz",
            "Experiment": "experiment",
            "Intensity": "intensity",
            "Score": "score",
            "Length": "reported_length",
            "Retention time": "retention_time",
            "CCS": "ccs",
        }
    )
    keep = [
        "sequence", "modified_sequence", "charge", "ccs", "mass", "precursor_mz",
        "experiment", "intensity", "score", "reported_length", "retention_time",
    ]
    df = add_canonical_fields(renamed[keep])
    df["source_dataset"] = "Meier_2021_official_SourceData_Figure_1"
    df["source_accessions"] = "PXD019086+PXD010012_official_merged_source_data"

    assert len(df) == 559_979
    assert df["precursor_key"].is_unique
    assert not df.isna().any().any()
    assert (df["intensity"] > 0).all() and (df["ccs"] > 0).all()
    assert df["charge"].between(2, 4).all()
    assert df["nAA"].between(7, 55).all()
    assert not df["experiment"].str.contains("PXD017703|diaPASEF", case=False, regex=True).any()

    counts = {
        "raw_observations_benchmark_user": 2_490_131,
        "official_notebook_filtered_endogenous_observations": OFFICIAL_PIPELINE_COUNTS["filtered_endogenous_observations"],
        "unique_sequence": int(df["sequence"].nunique()),
        "unique_modified_sequence": int(df["modified_sequence"].nunique()),
        "unique_sequence_charge": int(df[["sequence", "charge"]].drop_duplicates().shape[0]),
        "unique_modified_sequence_charge": int(df[["modified_sequence", "charge"]].drop_duplicates().shape[0]),
        "final_usable_unique_ccs": len(df),
    }
    return df, counts


def deterministic_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hashes = df["precursor_key"].map(
        lambda x: hashlib.sha256(f"{EXPERIMENT_ID}|validation|{x}".encode()).hexdigest()
    )
    order = np.argsort(hashes.to_numpy(), kind="stable")
    n_val = round(len(df) * 0.10)
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    validation = df.iloc[val_idx].copy().sort_values("precursor_key").reset_index(drop=True)
    training = df.iloc[train_idx].copy().sort_values("precursor_key").reset_index(drop=True)
    assert set(training.precursor_key).isdisjoint(validation.precursor_key)
    assert len(training) + len(validation) == len(df)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "split_key": "canonical modified-sequence+charge precursor_key",
        "algorithm": "lexicographically smallest SHA256(full_data_clean_v1|validation|precursor_key) to validation",
        "validation_fraction_target": 0.10,
        "training_pool_rows": len(training),
        "validation_rows": len(validation),
        "training_validation_overlap": 0,
        "proteometools_role": "fixed independent evaluation only; prohibited for early stopping/model selection/tuning",
        "validation_role": "early stopping and checkpoint selection only",
    }
    return training, validation, manifest


def build_proteometools(meier: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(FIG4)
    # Intentionally discard the original model's prediction column.
    pt = raw.rename(
        columns={"CCS": "ccs", "Charge": "charge", "Modified_sequence": "modified_sequence"}
    )[["modified_sequence", "charge", "ccs"]].copy()
    pt["sequence"] = pt["modified_sequence"].str.replace(r"\([^)]*\)|\[[^]]*\]|_", "", regex=True)
    pt = add_canonical_fields(pt)
    pt["source_dataset"] = "Meier_2021_official_SourceData_Figure_4_ProteomeTools"
    pt["source_accessions"] = "PXD019086_ProteomeTools"
    assert len(pt) == 154_235
    assert pt["precursor_key"].is_unique
    assert pt["ccs"].gt(0).all() and pt["charge"].between(2, 4).all()

    seq_overlap = len(set(pt.sequence) & set(meier.sequence))
    mod_overlap = len(set(pt.modified_sequence) & set(meier.modified_sequence))
    precursor_overlap = len(set(pt.precursor_key) & set(meier.precursor_key))
    stats = {
        "rows": len(pt),
        "unique_sequence": int(pt.sequence.nunique()),
        "unique_modified_sequence": int(pt.modified_sequence.nunique()),
        "sequence_overlap_with_meier": seq_overlap,
        "modified_sequence_overlap_with_meier": mod_overlap,
        "modified_sequence_charge_overlap_with_meier": precursor_overlap,
        "paper_target": OFFICIAL_PIPELINE_COUNTS["paper_independent_proteometools_test"],
        "official_figure4_difference_from_paper": len(pt) - OFFICIAL_PIPELINE_COUNTS["paper_independent_proteometools_test"],
        "freeze_status": "BLOCKED: official Figure-4 Source Data differs by 769 rows from paper n=155004 and contains 114832 unique plain sequences",
    }
    assert precursor_overlap == 0, f"ProteomeTools precursor overlap is {precursor_overlap}"
    return pt.sort_values("precursor_key").reset_index(drop=True), stats


def build_pxd017703(meier: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    usecols = [
        "R.FileName", "R.Replicate", "PEP.StrippedSequence", "EG.ModifiedSequence",
        "EG.IsDecoy", "FG.Charge", "FG.IonMobility", "FG.PrecMz",
    ]
    raw = pd.read_csv(PXD_REPORT, usecols=usecols, encoding="latin1", low_memory=False)
    raw = raw.rename(
        columns={
            "R.FileName": "run", "R.Replicate": "replicate", "PEP.StrippedSequence": "sequence",
            "EG.ModifiedSequence": "modified_sequence", "FG.Charge": "charge",
            "FG.IonMobility": "ion_mobility", "FG.PrecMz": "precursor_mz",
        }
    )
    decoy = raw["EG.IsDecoy"].astype(str).str.lower().isin(["true", "1", "+"])
    raw = raw.loc[~decoy].drop(columns=["EG.IsDecoy"])
    for col in ["charge", "ion_mobility", "precursor_mz"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["sequence", "modified_sequence", "charge", "ion_mobility", "precursor_mz"])
    raw = raw[
        raw["charge"].between(2, 6)
        & raw["ion_mobility"].between(0.3, 2.0)
        & raw["precursor_mz"].gt(0)
        & raw["sequence"].str.len().between(7, 55)
    ].copy()
    raw["charge"] = raw["charge"].astype(int)
    raw = add_canonical_fields(raw)
    raw["ccs"] = mobility_to_ccs_bruker(
        raw["ion_mobility"].to_numpy(np.float64),
        raw["charge"].to_numpy(np.float64),
        raw["precursor_mz"].to_numpy(np.float64),
    ).astype(np.float32)
    raw["source_dataset"] = "PXD017703_secondary_external_validation"
    raw["source_accession"] = "PXD017703"

    group = ["run", "replicate", "precursor_key", "sequence", "modified_sequence", "charge", "mods", "mod_sites", "nAA", "source_dataset", "source_accession"]
    all_df = (
        raw.groupby(group, as_index=False, dropna=False)
        .agg(ccs=("ccs", "median"), ion_mobility=("ion_mobility", "median"), precursor_mz=("precursor_mz", "median"), observations=("ccs", "size"))
        .sort_values(["run", "precursor_key"])
        .reset_index(drop=True)
    )
    meier_keys = set(meier.precursor_key)
    overlap_keys = set(all_df.precursor_key) & meier_keys
    unseen = all_df.loc[~all_df.precursor_key.isin(meier_keys)].copy().reset_index(drop=True)
    stats = {
        "raw_report_rows": int(len(pd.read_csv(PXD_REPORT, usecols=["R.FileName"], encoding="latin1"))),
        "filtered_observation_rows": len(raw),
        "dataset_run_precursor_rows": len(all_df),
        "unique_precursors": int(all_df.precursor_key.nunique()),
        "overlap_unique_precursors_with_meier": len(overlap_keys),
        "overlap_dataset_run_rows": int(all_df.precursor_key.isin(meier_keys).sum()),
        "unseen_dataset_run_rows": len(unseen),
        "unseen_unique_precursors": int(unseen.precursor_key.nunique()),
        "definition": "unseen means canonical modified-sequence+charge absent from the full Meier precursor pool",
    }
    return all_df, unseen, stats


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    assert_clean_training_provenance(TRAINING_PROVENANCE)

    meier, meier_counts = build_meier()
    training, validation, split_manifest = deterministic_split(meier)
    proteometools, pt_stats = build_proteometools(meier)
    pxd_all, pxd_unseen, pxd_stats = build_pxd017703(meier)

    outputs = {
        "full_dataset.parquet": meier.sort_values("precursor_key").reset_index(drop=True),
        "training_pool.parquet": training,
        "validation_set.parquet": validation,
        "proteometools_test.parquet": proteometools,
        "pxd017703_all.parquet": pxd_all,
        "pxd017703_unseen.parquet": pxd_unseen,
    }
    for name, frame in outputs.items():
        frame.to_parquet(ARTIFACTS / name, index=False)

    source_files = [FIG1, FIG4, ALIGNMENT_NOTEBOOK, PXD_REPORT]
    raw_archives = []
    for p in sorted(RAW_ARCHIVE_ROOT.glob("Results_*.zip")):
        raw_archives.append(
            {
                "path": str(p.resolve()),
                "bytes": p.stat().st_size,
                "evidence_data_rows": RAW_ARCHIVE_EVIDENCE_ROWS.get(p.name),
                "role": (
                    "EXCLUDED secondary validation"
                    if p.name == "Results_diaPASEF.zip"
                    else "existing MaxQuant evidence input; read-only; not the selected final training table"
                ),
            }
        )
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "clean_root": str(ROOT),
        "formal_training_started": False,
        "training_provenance_contamination_check": "PASS",
        "training_sources": TRAINING_PROVENANCE,
        "pxd017703_role": "secondary external validation only; excluded from every training/validation/sampling pool",
        "server_prepared_asset_priority_audit": {
            "priority_root": str(RAW_ARCHIVE_ROOT),
            "detailed_archive_inventory": str(ARTIFACTS / "existing_server_asset_audit.json"),
            "audit_mode": "read-only inventory before any download or reprocessing",
            "files_inspected": "all files recursively plus every member of Results_*.zip",
            "direct_559979_row_processed_dataset_found": False,
            "finding": "Organism-level and ProteomeTools Results archives contain evidence.txt, but no aligned/combined/final CCS table; Results_diaPASEF.zip is external-only.",
            "existing_endogenous_evidence_rows_in_six_PXD019086_archives": 1_882_776,
            "existing_proteometools_evidence_rows_in_three_archives": 330_696,
            "exact_file_currently_producing_verified_559979_dataset": str(FIG1),
            "exact_file_sha256": sha256_file(FIG1),
            "decision": "Reuse verified final Source Data for clean retraining; do not rebuild raw PXD019086/PXD010012 data.",
            "new_download_or_raw_reprocessing_required": False,
            "training_goal": "retrain models from random initialization, not rebuild raw data",
        },
        "source_files": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in source_files
        ],
        "server_results_zip_inventory": raw_archives,
        "paper": {
            "title": "Deep learning the collisional cross sections of the peptide universe from a million experimental values",
            "doi": "10.1038/s41467-021-21352-8",
            "primary_final_data": "official Nature Source Data Figure 1",
            "accessions": ["PXD019086", "PXD010012"],
        },
        "official_pipeline_counts": OFFICIAL_PIPELINE_COUNTS,
        "meier_counts": meier_counts,
        "proteometools": pt_stats,
        "pxd017703": pxd_stats,
        "raw_reconstruction_readiness": {
            "PXD019086_alone_contains_all_endogenous_input": False,
            "PXD010012_required_for_methods_level_raw_reconstruction": True,
            "official_final_source_data_available": True,
            "server_approximately_2_4M_interpretation": "2,465,262 is the official notebook's post-filter endogenous evidence count, not proof that any arbitrary ~2.4M-row file is complete",
        },
    }

    split_manifest["artifact_sha256"] = {name: sha256_file(ARTIFACTS / name) for name in outputs}
    split_manifest["counts"] = {name: len(frame) for name, frame in outputs.items()}
    (ARTIFACTS / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n")
    (ARTIFACTS / "data_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")

    # Final contamination assertion checks only training-bearing artifacts and provenance.
    assert_clean_training_provenance(TRAINING_PROVENANCE + meier["source_dataset"].unique().tolist() + meier["source_accessions"].unique().tolist())
    print(json.dumps({"meier": meier_counts, "proteometools": pt_stats, "pxd017703": pxd_stats, "split": split_manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
