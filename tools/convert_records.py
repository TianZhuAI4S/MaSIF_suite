"""One-time TFRecord → numeric NPZ conversion; run only in masif_nn."""

import argparse
import json
import re
from pathlib import Path
import numpy as np
import tensorflow as tf


def decode_record(serialized):
    """Decode the original TF Example shapes and values with no preprocessing changes."""
    example = tf.train.Example.FromString(serialized)
    fields = example.features.feature
    output = {}
    for source, target, kind in (("input_feat", "features", "float"),
                                  ("rho_wrt_center", "rho", "float"),
                                  ("theta_wrt_center", "theta", "float"),
                                  ("mask", "mask", "float"),
                                  ("pocket_labels", "pocket_labels", "int64")):
        shape = list(fields[source + "_shape"].int64_list.value)
        values = getattr(fields[source], kind + "_list").value
        output[target] = np.asarray(values, dtype=np.float32 if kind == "float" else np.int64).reshape(shape)
    output["pdb"] = np.asarray(fields["pdb"].bytes_list.value[0].decode("utf-8"))
    return output


def convert_records(source, output):
    """Write safe structure packets and verify every stored array exactly matches TF."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = set()
    rows = []
    for serialized in tf.compat.v1.io.tf_record_iterator(str(source)):
        record = decode_record(serialized)
        name = str(record["pdb"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError(f"Unsafe or duplicate PDB record name: {name}")
        names.add(name)
        destination = output / f"{name}.npz"
        np.savez_compressed(destination, **record)
        with np.load(destination, allow_pickle=False) as stored:
            for key, value in record.items():
                if not np.array_equal(value, stored[key]):
                    raise AssertionError(f"TFRecord conversion changed {name}/{key}")
        rows.append(dict(pdb=name, patches=int(record["features"].shape[0]),
                         pockets=int(record["pocket_labels"].shape[1])))
    report = dict(source=str(source), tensorflow=tf.__version__, exact_array_roundtrip=True, records=rows)
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Converted {len(rows)} structures; every array round-tripped exactly")
    return report


def main():
    """Convert one original ligand train/validation/test TFRecord file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert_records(args.input, args.output)


if __name__ == "__main__":
    main()
