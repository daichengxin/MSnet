import os

import duckdb


def detect_parquet_schema(parquet_paths) -> str:
    """Return ``"legacy"`` or ``"current"`` for the MSNet parquet column layout.

    The quantms/msnet collection was re-processed with a new layout
    (``charge``, ``observed_mz``, ``calculated_mz``, ``rt``,
    ``run_file_name`` and a ``(cv_name, cv_value)[]`` ``cv_params`` array);
    files generated before the re-processing use ``precursor_charge``,
    ``exp_mass_to_charge``, ``retention_time``, ``reference_file_name`` and a
    named-struct ``cv_params``. Only the first input file is inspected, so all
    files of one dataset must share a layout.

    Parameters
    ----------
    parquet_paths:
        One or more ``*-MSNet.parquet`` files.

    Returns
    -------
    str
        ``"legacy"`` or ``"current"``.
    """
    if isinstance(parquet_paths, (str, os.PathLike)):
        first = str(parquet_paths)
    else:
        first = str(next(iter(parquet_paths)))
    con = duckdb.connect()
    columns = [
        row[0] for row in con.execute("DESCRIBE SELECT * FROM parquet_scan(?)", [first]).fetchall()
    ]
    if "precursor_charge" in columns:
        return "legacy"
    if "charge" in columns:
        return "current"
    raise ValueError(
        f"Unrecognised MSNet parquet layout in {first}: "
        f"expected a 'precursor_charge' or 'charge' column, found {columns}"
    )


def dereduant_precursor(cursor, key="peptidoform"):
    """De-duplicate the rows of an open duckdb cursor on *key*.

    Keeps the first row for every distinct value of *key* and returns the
    remaining rows in their original order.

    Parameters
    ----------
    cursor:
        An open duckdb query result whose columns include *key*.
    key:
        Column to de-duplicate on.

    Returns
    -------
    list[tuple]
        Rows with only the first occurrence kept per distinct *key* value.
    """
    columns = [description[0] for description in cursor.description or []]
    if key not in columns:
        raise ValueError(f"cursor result has no {key!r} column; available columns: {columns}")

    key_index = columns.index(key)
    seen = set()
    output_psms = []
    for row in cursor.fetchall():
        value = row[key_index]
        if value in seen:
            continue
        seen.add(value)
        output_psms.append(row)
    return output_psms
