import torch
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from torch.utils.data import Sampler
import math
import random
import re


class MS2TorchDataset(TorchDataset):
    def __init__(self, precursor_df, fragment_df, feature_columns=None, label_column="fragments"):
        self.precursor_df = precursor_df
        self.fragment_df = fragment_df
        self.feature_columns = feature_columns or [
            "sequence",
            "charge",
            "mods",
            "mod_sites",
            "nce",
            "nAA",
            "instrument",
        ]
        self.label_column = label_column

    def __len__(self):
        return len(self.precursor_df)

    def __getitem__(self, idx):

        row = self.precursor_df.iloc[idx]

        features = {k: row[k] for k in self.feature_columns}

        # slice fragment（关键）
        start = row["frag_start_idx"]
        stop = row["frag_stop_idx"]

        fragments = self.fragment_df.iloc[start:stop].values

        label = torch.tensor(fragments, dtype=torch.float32)

        return features, label


class LengthExactSampler(Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.groups = {}

        for idx, l in enumerate(lengths):
            if l not in self.groups:
                self.groups[l] = []
            self.groups[l].append(idx)

    def __iter__(self):

        batches = []

        for length_value, indices in self.groups.items():

            if self.shuffle:
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]

                if len(batch) > 0:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return sum(
            math.ceil(len(indices)/self.batch_size)
            for indices in self.groups.values()
        )


def parse_nce_fast(nce_raw):
    if nce_raw is None:
        return None
    try:
        return float(nce_raw)
    except:
        m = re.search(r"\d+\.?\d*", str(nce_raw))
        return float(m.group()) if m else None


def format_modifications_custom(mods_raw, sequence):

    if not mods_raw:
        return "", ""

    seq_len = len(sequence)

    mods_list = []
    sites_list = []

    for mod in mods_raw:

        if not mod or "name" not in mod:
            continue

        name = mod["name"].split(" ")[0]

        if "positions" not in mod or not mod["positions"]:
            continue

        for pos_info in mod["positions"]:

            if not pos_info or "position" not in pos_info:
                continue

            position = int(pos_info["position"])
            # N-term
            if position == 0:
                site = "Protein_N-term"
            # C-term
            elif position == seq_len + 1:
                site = "Protein_C-term"
            # AA
            elif 1 <= position <= seq_len:
                site = sequence[position - 1]
            else:
                continue

            # 按你要求的格式
            mods_list.append(f"{name}@{site}")
            sites_list.append(str(position))

    return ";".join(mods_list), ";".join(sites_list)


class AlphaPeptDeepConverter:
    def __init__(self, parquet_path, batch_size=100_000):
        self.parquet_path = parquet_path
        self.batch_size = batch_size

    def convert_parquet_to_training_format(self):

        parquet_file = pq.ParquetFile(self.parquet_path)

        precursor_rows = []
        fragment_rows = []

        global_frag_idx = 0

        for batch in parquet_file.iter_batches(batch_size=self.batch_size):

            table = pa.Table.from_batches([batch])

            sequences = table["sequence"]
            charges = table["precursor_charge"]
            modifications = table["modifications"]

            ion_types = table["ion_type_array"]
            charge_array = table["charge_array"]
            intensity_array = table["intensity_array"]

            cv_struct = table["cv_params"]

            # 批量抽取 struct 字段（零拷贝）
            instrument_array = pa.chunked_array(
                [chunk.field("Instrument") for chunk in cv_struct.chunks]
            )
            nce_array = pa.chunked_array(
                [chunk.field("Collision Energy") for chunk in cv_struct.chunks]
            )

            n_rows = table.num_rows

            for i in range(n_rows):

                seq = sequences[i].as_py()
                z = charges[i].as_py()

                # --------------------
                # modifications
                # --------------------
                mods = ""
                mod_sites = ""

                mods_raw = modifications[i].as_py()

                if mods_raw:

                    mods, mod_sites = format_modifications_custom(mods_raw, seq)

                # ----------------------------
                # instrument + NCE
                # ----------------------------
                instrument_val = instrument_array[i]
                instrument = instrument_val.as_py() if instrument_val else None

                nce_val = nce_array[i]
                nce = parse_nce_fast(nce_val.as_py()) if nce_val else None

                # --------------------
                # fragment
                # --------------------
                ions = ion_types[i].as_py()
                frag_charges = charge_array[i].as_py()
                intensities = intensity_array[i].as_py()

                frag_start = global_frag_idx

                seq_len = len(seq)

                # 构建映射: (ion_type, position, charge) -> intensity
                frag_dict = {}
                max_intensity = 0
                for ion, frag_z, inten in zip(ions, frag_charges, intensities):

                    if ion is None:
                        continue

                    # 忽略中性丢失
                    if "-" in ion:
                        continue

                    # 只接受 b/y
                    ion_type = ion[0]

                    if ion_type not in ("b", "y"):
                        continue

                    # 提取数字
                    try:
                        position = int(ion[1:])
                    except:
                        continue

                    if inten > max_intensity:
                        max_intensity = inten
                    frag_dict[(ion_type, position, frag_z)] = inten

                frag_start = global_frag_idx

                # 对每个 cleavage 位点生成一行
                # b ions: 1 → n-1
                # y ions: 1 → n-1
                for pos in range(1, seq_len):
                    b_z1 = frag_dict.get(("b", pos, 1), 0.0) / max_intensity
                    b_z2 = frag_dict.get(("b", pos, 2), 0.0) / max_intensity
                    y_z1 = frag_dict.get(("y", pos, 1), 0.0) / max_intensity
                    y_z2 = frag_dict.get(("y", pos, 2), 0.0) / max_intensity

                    fragment_rows.append((b_z1, b_z2, y_z1, y_z2))
                    global_frag_idx += 1

                frag_stop = global_frag_idx
                # print(mods)
                # print(mod_sites)
                precursor_rows.append(
                    (
                        seq,
                        z,
                        mods,
                        mod_sites,
                        len(seq),
                        frag_stop,
                        frag_start,
                        nce,
                        instrument,
                    )
                )

        # ----------------------------
        # 加列名
        # ----------------------------

        precursor_df = pd.DataFrame(
            precursor_rows,
            columns=[
                "sequence",
                "charge",
                "mods",
                "mod_sites",
                "nAA",
                "frag_stop_idx",
                "frag_start_idx",
                "nce",
                "instrument",
            ],
        )

        fragment_df = pd.DataFrame(
            fragment_rows,
            columns=[
                "b_z1",
                "b_z2",
                "y_z1",
                "y_z2",
            ],
        )

        return precursor_df, fragment_df