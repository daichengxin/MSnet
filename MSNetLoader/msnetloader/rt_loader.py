import torch
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from torch.utils.data import IterableDataset


class RTTorchDataset(TorchDataset):

    def __init__(
        self,
        parquet_path,
        label_column="retention_time",
    ):
        table = pq.read_table(parquet_path)

        # ------------------------
        # load columns
        # ------------------------
        seqs = table["sequence"].to_pylist()
        mods = table["modifications"].to_pylist()
        labels = np.array(table[label_column], dtype=np.float32)

        # ------------------------
        # format modifications
        # ------------------------
        mods_fmt = []
        for m, s in zip(mods, seqs):
            if m:
                mods_fmt.append(
                    self.format_modifications_ms2pip_style(m, s)
                )
            else:
                mods_fmt.append("")

        # ------------------------
        # nAA + sort
        # ------------------------
        nAA = np.array([len(s) for s in seqs], dtype=np.int32)
        order = np.argsort(nAA)

        self.features = {
            "sequence": [seqs[i] for i in order],
            "modifications": [mods_fmt[i] for i in order]
        }

        self.labels = labels[order] / 60.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        features = {
            "sequence": self.features["sequence"][idx],
            "modifications": self.features["modifications"][idx]
        }

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return features, label

    @staticmethod
    def format_modifications_ms2pip_style(mods_raw, seq):

        formatted = []

        for mod in mods_raw or []:
            name = mod.get("name")
            if not name:
                continue

            for p in mod.get("positions", []):
                pos = p.get("position")
                if pos is not None:
                    formatted.append(f"{pos}|{name}")

        return "|".join(formatted)


class RTIterableDataset(IterableDataset):

    def __init__(self, parquet_path, batch_size=100_000):
        self.parquet_path = parquet_path
        self.batch_size = batch_size

    def __iter__(self):

        parquet_file = pq.ParquetFile(self.parquet_path)

        for batch in parquet_file.iter_batches(batch_size=self.batch_size):

            table = pa.Table.from_batches([batch])

            sequences = table["sequence"]
            modifications = table["modifications"]
            tr_array = table["retention_time"]

            for i in range(table.num_rows):

                seq = sequences[i].as_py()
                tr = tr_array[i].as_py()

                mods_raw = modifications[i].as_py()
                mods_fmt = ""

                if mods_raw:
                    mods_fmt = RTTorchDataset.format_modifications_ms2pip_style(
                        mods_raw, seq
                    )

                yield {
                    "sequence": seq,
                    "modifications": mods_fmt
                }, torch.tensor(tr / 60.0, dtype=torch.float32)


class DeepLCConverter:

    def __init__(self, parquet_path, batch_size=100_000):
        self.parquet_path = parquet_path
        self.batch_size = batch_size
    
    @staticmethod
    def format_modifications_ms2pip_style(mods_raw, seq):

        formatted = []

        for mod in mods_raw:

            if not mod:
                continue

            name = mod.get("name")

            if not name:
                continue

            positions = mod.get("positions", [])

            for p in positions:

                pos = p.get("position")

                if pos is None:
                    continue

                # ------------------------
                # 位置规则
                # ------------------------
                # 如果你 parquet 已经是 1-based，直接用
                # 如果是 0-based，需要 +1

                formatted.append(f"{pos}|{name}")

        return "|".join(formatted)
    
    def convert_parquet_to_rt_format(self):

        parquet_file = pq.ParquetFile(self.parquet_path)

        rows = []

        for batch in parquet_file.iter_batches(batch_size=self.batch_size):

            table = pa.Table.from_batches([batch])

            sequences = table["sequence"]
            modifications = table["modifications"]

            # RT 字段（根据你 parquet 实际字段名改）
            tr_array = table["retention_time"]

            n_rows = table.num_rows

            for i in range(n_rows):

                seq = sequences[i].as_py()

                # --------------------
                # RT
                # --------------------
                tr = tr_array[i].as_py()

                # --------------------
                # modifications
                # --------------------
                mods_formatted = ""

                mods_raw = modifications[i].as_py()

                if mods_raw:
                    mods_formatted = self.format_modifications_ms2pip_style(
                        mods_raw,
                        seq
                    )

                rows.append(
                    (
                        seq,
                        mods_formatted,
                        float(tr) / 60
                    )
                )

        df = pd.DataFrame(
            rows,
            columns=[
                "seq",
                "modifications",
                "tr"
            ],
        )

        return df
