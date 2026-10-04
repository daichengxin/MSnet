"""Loader and split tests against the current (re-processed) MSNet schema.

The quantms/msnet collection moved to a new column layout (``charge``,
``observed_mz``, ``rt`` and a ``(cv_name, cv_value)[]`` ``cv_params`` array);
these tests pin that the loaders handle it, using a small slice of the real
``PXD014877-Sulfolobus-solfataricus`` file as fixture.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from msnetloader.denovo_loader import DeNovoIterableDataset
from msnetloader.ms2_loader import MS2TorchDataset
from msnetloader.rt_loader import RTIterableDataset
from msnetloader.split import split_dataset, split_peptides
from msnetloader.utils import detect_parquet_schema

TESTS_DIR = Path(__file__).parent
CURRENT_FILE = str(TESTS_DIR / "test_data/PXD014877-Sulfolobus_solfataricus-MSNet.parquet")


def test_detects_current_schema():
    assert detect_parquet_schema(CURRENT_FILE) == "current"


def test_ms2_dataset_and_dataloader():
    dataset = MS2TorchDataset([CURRENT_FILE], ion_types=("b", "y"))
    batch = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))

    assert isinstance(batch, dict)
    assert batch["targets"].numel() > 0
    assert torch.all(batch["charge"] >= 1)
    # instrument and nce come from the (cv_name, cv_value)[] cv_params array
    assert batch["instruments"]
    assert batch["nce"].numel() > 0
    assert torch.all(batch["nce"] > 0)


def test_ms2_loader_filters():
    dataset = MS2TorchDataset([CURRENT_FILE], min_consensus_support=0.5, max_pep=0.05)
    batch = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))
    assert batch["peptide"]


def test_denovo_dataloader():
    dataset = DeNovoIterableDataset([CURRENT_FILE])
    batch = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))

    assert len(batch["sequence"]) > 0
    assert batch["precursor_mz"].numel() > 0
    assert torch.all(batch["charge"] >= 1)


def test_rt_dataloader():
    dataset = RTIterableDataset([CURRENT_FILE], batch_size=1000, min_consensus_support=0.0, max_pep=1.0)
    batch = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))

    assert len(batch["peptide"]) > 0
    rt = batch["rt"]
    assert max(rt) > 0
    # rt is stored in seconds and converted to minutes by the loader
    assert max(rt) <= 200


def test_split_peptides():
    splits = split_peptides([CURRENT_FILE], test_size=0.2, seed=7)
    assert splits["train"] and splits["test"]


def test_split_dataset_roundtrip(tmp_path):
    splits = split_dataset([CURRENT_FILE], output_dir=tmp_path, test_size=0.2, seed=7)
    for paths in splits.values():
        assert paths[0].stat().st_size > 0

    outputs = [str(p) for p in tmp_path.glob("*.parquet")]
    dataset = MS2TorchDataset(outputs)
    batch = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))
    assert batch["targets"].numel() > 0


def test_tf_variants_load_current_schema():
    tf = pytest.importorskip("tensorflow")

    from msnetloader.denovo_tf import DeNovoTFDataset
    from msnetloader.ms2_tf import MS2TFDataset
    from msnetloader.rt_tf import RTTFDataset

    ds = MS2TFDataset([CURRENT_FILE], batch_size=16)
    batch = next(iter(ds.generator()))
    assert batch["targets"].shape[-1] > 0
    assert len(batch["instruments"]) > 0

    ds = DeNovoTFDataset([CURRENT_FILE], batch_size=16)
    rows = list(ds.generator())
    assert rows, "de novo TF dataset produced no rows"

    ds = RTTFDataset([CURRENT_FILE], batch_size=1000)
    batch = next(iter(ds.generator()))
    assert len(batch["peptide"]) > 0

    assert tf is not None
