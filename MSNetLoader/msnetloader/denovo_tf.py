import tensorflow as tf
import pyarrow.parquet as pq
import numpy as np
import pyarrow as pa

def get_output_signature():
    return {
        "spectrum": tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        "sequence": tf.TensorSpec(shape=(), dtype=tf.string),
        "precursor_mz": tf.TensorSpec(shape=(), dtype=tf.float32),
        "charge": tf.TensorSpec(shape=(), dtype=tf.int32),
    }


def build_dataset(ds, batch_size):

    ds = ds.padded_batch(
        batch_size,
        padded_shapes={
            "spectrum": [None, 2],   # 自动 padding
            "sequence": [],
            "precursor_mz": [],
            "charge": [],
        },
        padding_values={
            "spectrum": tf.constant(0, tf.float32),
            "sequence": tf.constant(b"", tf.string),
            "precursor_mz": tf.constant(0, tf.float32),
            "charge": tf.constant(0, tf.int32),
        }
    )

    return ds.prefetch(tf.data.AUTOTUNE)

class TFDeNovoDataset:

    def __init__(self, data_path, max_peaks=150):
        self.parquet = pq.ParquetFile(data_path)
        self.max_peaks = max_peaks
        self.length = self.parquet.metadata.num_rows
        self.row_group_size = self.parquet.metadata.row_group(0).num_rows

    def generator(self):

        for idx in range(self.length):

            group = idx // self.row_group_size
            offset = idx % self.row_group_size

            table = self.parquet.read_row_group(group)

            mz = np.asarray(table["mz_array"][offset].as_py())
            intensity = np.asarray(table["intensity_array"][offset].as_py())

            sequence = table["sequence"][offset].as_py()
            charge = table["precursor_charge"][offset].as_py()
            precursor_mz = table["exp_mass_to_charge"][offset].as_py()

            # top peaks
            if len(mz) > self.max_peaks:
                idxs = np.argpartition(intensity, -self.max_peaks)[-self.max_peaks:]
                mz = mz[idxs]
                intensity = intensity[idxs]

            if len(intensity) > 0 and intensity.max() > 0:
                intensity = intensity / intensity.max()

            spectrum = np.stack([mz, intensity], axis=1).astype(np.float32)

            yield {
                "spectrum": spectrum,
                "sequence": sequence.encode(),
                "precursor_mz": np.float32(precursor_mz),
                "charge": np.int32(charge),
            }

    def get_dataset(self):

        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=get_output_signature()
        )

        return ds

    class TFDeNovoIterableDataset:

        def __init__(self, data_path, max_peaks=150, batch_size=100_000):
            self.data_path = data_path
            self.max_peaks = max_peaks
            self.batch_size = batch_size

        def generator(self):

            parquet_file = pq.ParquetFile(self.data_path)

            for batch in parquet_file.iter_batches(batch_size=self.batch_size):

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

                    # top peaks
                    if len(mz) > self.max_peaks:
                        idxs = np.argpartition(intensity, -self.max_peaks)[-self.max_peaks:]
                        mz = mz[idxs]
                        intensity = intensity[idxs]

                    if len(intensity) > 0 and intensity.max() > 0:
                        intensity = intensity / intensity.max()

                    spectrum = np.stack([mz, intensity], axis=1).astype(np.float32)

                    yield {
                        "spectrum": spectrum,
                        "sequence": sequence.encode(),
                        "precursor_mz": np.float32(precursor_mz),
                        "charge": np.int32(charge),
                    }

        def get_dataset(self):

            ds = tf.data.Dataset.from_generator(
                self.generator,
                output_signature=get_output_signature()
            )

            return ds


# dataset = TFDeNovoIterableDataset("data.parquet")
#
# ds = dataset.get_dataset()
#
# ds = build_dataset(ds, batch_size=32)
#
# for batch in ds:
#     print(batch["spectrum"].shape)