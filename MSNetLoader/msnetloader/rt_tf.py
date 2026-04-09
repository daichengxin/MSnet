import tensorflow as tf
import pyarrow.parquet as pq
import numpy as np
from rt_loader import RTTorchDataset
import pyarrow as pa


class RTDatasetTF:

    def __init__(self, parquet_path, label_column="retention_time"):

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
        # sort by length
        # ------------------------
        nAA = np.array([len(s) for s in seqs], dtype=np.int32)
        order = np.argsort(nAA)

        self.seqs = [seqs[i] for i in order]
        self.mods = [mods_fmt[i] for i in order]
        self.labels = labels[order] / 60.0

    def to_tf_dataset(self, batch_size=32, shuffle=False):

        ds = tf.data.Dataset.from_tensor_slices((
            {
                "sequence": self.seqs,
                "modifications": self.mods
            },
            self.labels
        ))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(self.labels))

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return ds

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


class RTIterableDatasetTF:

    def __init__(self, parquet_path, batch_size=100_000):
        self.parquet_path = parquet_path
        self.batch_size = batch_size

    def generator(self):

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
                }, np.float32(tr / 60.0)

    def to_tf_dataset(self, batch_size=32):

        output_signature = (
            {
                "sequence": tf.TensorSpec(shape=(), dtype=tf.string),
                "modifications": tf.TensorSpec(shape=(), dtype=tf.string),
            },
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )

        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=output_signature
        )

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return ds