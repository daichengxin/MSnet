import torch
from torch.utils.data import Dataset as TorchDataset
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from torch.utils.data import IterableDataset


def denovo_collate_fn(batch):

    spectra = [b["spectrum"] for b in batch]
    tokens = [b["sequence"] for b in batch]

    precursor_mz = torch.stack([b["precursor_mz"] for b in batch])
    charge = torch.stack([b["charge"] for b in batch])

    # -----------------
    # spectrum padding
    # -----------------

    max_peaks = max(s.shape[0] for s in spectra)

    padded_spec = []

    for s in spectra:

        pad_len = max_peaks - s.shape[0]

        if pad_len > 0:
            pad = torch.zeros(pad_len, 2)
            s = torch.cat([s, pad], dim=0)

        padded_spec.append(s)

    spectra = torch.stack(padded_spec)

    return {
        "spectra": spectra,
        "precursor_mz": precursor_mz,
        "charge": charge,
        "tokens": tokens
    }


class PeptideTokenizer:

    def __init__(self):

        aas = list("ACDEFGHIKLMNPQRSTVWY")

        self.pad = "<pad>"
        self.bos = "<bos>"
        self.eos = "<eos>"

        self.vocab = [self.pad, self.bos, self.eos] + aas

        self.stoi = {a:i for i,a in enumerate(self.vocab)}
        self.itos = {i:a for i,a in enumerate(self.vocab)}

        self.pad_id = self.stoi[self.pad]
        self.bos_id = self.stoi[self.bos]
        self.eos_id = self.stoi[self.eos]

    def encode(self, seq):

        tokens = [self.bos_id]

        for a in seq:
            tokens.append(self.stoi.get(a, self.pad_id))

        tokens.append(self.eos_id)

        return tokens


class DeNovoDataset(TorchDataset):

    def __init__(self, data_path, max_peaks=150):

        self.max_peaks = max_peaks

        self.parquet = pq.ParquetFile(data_path)
        self.backend = "parquet"
        self.length = self.parquet.metadata.num_rows

        # row group size
        self.row_group_size = self.parquet.metadata.row_group(0).num_rows

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        group = idx // self.row_group_size
        offset = idx % self.row_group_size

        table = self.parquet.read_row_group(group)

        mz = table["mz_array"][offset].as_py()
        intensity = table["intensity_array"][offset].as_py()
        sequence = table["sequence"][offset].as_py()
        charge = table["precursor_charge"][offset].as_py()
        precursor_mz = table["exp_mass_to_charge"][offset].as_py()

        # numpy
        mz = np.asarray(mz)
        intensity = np.asarray(intensity)

        # top peaks
        if len(mz) > self.max_peaks:

            idxs = np.argsort(intensity)[-self.max_peaks:]
            mz = mz[idxs]
            intensity = intensity[idxs]

        # normalize
        if intensity.max() > 0:
            intensity = intensity / intensity.max()

        spectrum = np.stack([mz, intensity], axis=1)

        spectrum = torch.tensor(spectrum, dtype=torch.float32)

        # tokens = torch.tensor(sequence, dtype=torch.long)

        precursor_mz = torch.tensor(precursor_mz, dtype=torch.float32)
        charge = torch.tensor(charge, dtype=torch.long)

        return {
            "spectrum": spectrum,
            "sequence": sequence,
            "precursor_mz": precursor_mz,
            "charge": charge
        }


class DeNovoIterableDataset(IterableDataset):

    def __init__(self, data_path, max_peaks=150, batch_size=100_000):
        self.data_path = data_path
        self.max_peaks = max_peaks
        self.batch_size = batch_size

    def __iter__(self):
        self.parquet_file = pq.ParquetFile(self.data_path)

        for batch in self.parquet_file.iter_batches(batch_size=self.batch_size):

            table = pa.Table.from_batches([batch])

            mz_array = table["mz_array"]
            intensity_array = table["intensity_array"]
            sequence_array = table["sequence"]
            charge_array = table["precursor_charge"]
            precursor_array = table["exp_mass_to_charge"]

            for i in range(table.num_rows):

                mz = np.asarray(mz_array[i].as_py())
                intensity = np.asarray(intensity_array[i].as_py())

                sequence = sequence_array[i].as_py()
                charge = charge_array[i].as_py()
                precursor_mz = precursor_array[i].as_py()

                # ------------------------
                # top peaks
                # ------------------------
                if len(mz) > self.max_peaks:
                    idxs = np.argsort(intensity)[-self.max_peaks:]
                    mz = mz[idxs]
                    intensity = intensity[idxs]

                # normalize
                if len(intensity) > 0 and intensity.max() > 0:
                    intensity = intensity / intensity.max()

                spectrum = np.stack([mz, intensity], axis=1)

                yield {
                    "spectrum": torch.tensor(spectrum, dtype=torch.float32),
                    "sequence": sequence,
                    "precursor_mz": torch.tensor(precursor_mz, dtype=torch.float32),
                    "charge": torch.tensor(charge, dtype=torch.long),
                }