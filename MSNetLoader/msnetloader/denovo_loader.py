import torch
import numpy as np
from torch.utils.data import IterableDataset
import duckdb


class DeNovoIterableDataset(IterableDataset):

    def __init__(self, parquet_path, max_peaks=150, batch_size=32,
                 min_consensus_support=None,
                 max_pep=None):
        con = duckdb.connect()

        query = f"""
            SELECT
                peptidoform,
                exp_mass_to_charge AS precursor_mz,
                precursor_charge AS charge,
                mz_array,
                consensus_support,
                posterior_error_probability,
                intensity_array
            FROM parquet_scan('{parquet_path}')
        """

        self.arrow_table = con.execute(query).to_arrow_table()
        self.batch_size = batch_size
        self.max_peaks = max_peaks
        self.batch_size = batch_size
        self.min_consensus_support = min_consensus_support
        self.max_pep = max_pep

    def __iter__(self):
        for batch in self.arrow_table.to_batches(max_chunksize=self.batch_size):
            yield self.process_batch(batch)

    # -----------------------------
    def process_batch(self, batch):

        peptidoform = batch["peptidoform"].to_pylist()
        charges = batch["charge"].to_pylist()
        mz_list = batch["mz_array"].to_pylist()
        int_list = batch["intensity_array"].to_pylist()
        precursor_mz = batch["precursor_mz"].to_pylist()
        consensus_supports = batch["consensus_support"].to_pylist()
        peps = batch["posterior_error_probability"].to_pylist()

        spectra_out = []
        seq_out = []
        charge_out = []
        precursor_out = []

        # -----------------------------
        # per spectrum processing
        # -----------------------------
        for i in range(len(peptidoform)):

            if not self.filter_by_consensus_support(consensus_supports[i]):
                continue

            if not self.filter_by_pep(peps[i]):
                continue

            mz = np.asarray(mz_list[i], dtype=np.float32)
            intensity = np.asarray(int_list[i], dtype=np.float32)

            if len(mz) == 0:
                continue

            # -------------------------
            # top-k peaks
            # -------------------------
            if len(mz) > self.max_peaks:
                idx = np.argsort(intensity)[-self.max_peaks:]
                mz = mz[idx]
                intensity = intensity[idx]

            # -------------------------
            # normalize
            # -------------------------
            max_int = intensity.max() if len(intensity) > 0 else 1.0
            intensity = intensity / max_int

            spectrum = np.stack([mz, intensity], axis=1)

            spectra_out.append(torch.tensor(spectrum, dtype=torch.float32))
            seq_out.append(peptidoform[i])
            charge_out.append(charges[i])
            precursor_out.append(precursor_mz[i])

        return {
            "spectrum": spectra_out,
            "sequence": seq_out,
            "precursor_mz": torch.tensor(precursor_out, dtype=torch.float32),
            "charge": torch.tensor(charge_out, dtype=torch.long),
        }

    # =========================================================
    # Optional FILTER 1
    # =========================================================
    def filter_by_consensus_support(self, support):
        """
        Keep spectrum if consensus_support >= threshold
        """
        if self.min_consensus_support is None:
            return True

        if support is None:
            return False

        return support >= self.min_consensus_support

    # =========================================================
    # Optional FILTER 2
    # =========================================================
    def filter_by_pep(self, pep):
        """
        Keep spectrum if posterior_error_probability <= threshold
        """
        if self.max_pep is None:
            return True

        if pep is None:
            return False

        return pep <= self.max_pep
