import tensorflow as tf
import pyarrow.parquet as pq
import numpy as np


class TFMSDataset:

    def __init__(self, parquet_path, batch_size=32):

        self.parquet_path = parquet_path
        self.batch_size = batch_size

        # load parquet
        table = pq.read_table(parquet_path)

        # to numpy
        self.data = {
            "sequence": table["sequence"].to_pylist(),
            "charge": np.array(table["charge"]),
            "mods": table["modifications"].to_pylist(),
            "mod_sites": table["mod_sites"].to_pylist(),
            "nce": np.array(table["nce"], dtype=np.float32),
            "instrument": table["instrument"].to_pylist(),
            "fragments": table["fragments"].to_pylist(),
        }

        # 🔥 compute nAA（和 torch 一样）
        self.data["nAA"] = np.array(
            [len(seq) for seq in self.data["sequence"]],
            dtype=np.int32
        )

        order = np.argsort(self.data["nAA"])

        for k in self.data:
            self.data[k] = [self.data[k][i] for i in order]

        self.size = len(self.data["sequence"])

    def generator(self):

        for i in range(self.size):

            features = {
                "sequence": self.data["sequence"][i].encode(),
                "charge": self.data["charge"][i],
                "mods": (self.data["mods"][i] or "").encode(),
                "mod_sites": (self.data["mod_sites"][i] or "").encode(),
                "nce": self.data["nce"][i],
                "nAA": self.data["nAA"][i],
                "instrument": (self.data["instrument"][i] or "").encode(),
            }

            fragments = np.array(
                self.data["fragments"][i],
                dtype=np.float32
            )

            yield features, fragments

    def get_dataset(self):

        output_signature = (
            {
                "sequence": tf.TensorSpec([], tf.string),
                "charge": tf.TensorSpec([], tf.int32),
                "mods": tf.TensorSpec([], tf.string),
                "mod_sites": tf.TensorSpec([], tf.string),
                "nce": tf.TensorSpec([], tf.float32),
                "nAA": tf.TensorSpec([], tf.int32),
                "instrument": tf.TensorSpec([], tf.string),
            },
            tf.TensorSpec([None, 4], tf.float32),
        )

        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=output_signature
        )

        ds = ds.batch(self.batch_size)

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds