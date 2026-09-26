from __future__ import annotations

import argparse
import random
from pathlib import Path

from rt_utils import deduplicate, load_grouped_pools, read_config, row_key, safe_name, stable_seed, write_rows, write_tasks


def balanced_nested_pool(groups: dict[str, list[dict]], seed: int, split_name: str) -> list[dict]:
    shuffled = {}
    for name, rows in sorted(groups.items()):
        items = list(rows)
        random.Random(stable_seed("train", seed, split_name, name)).shuffle(items)
        shuffled[name] = items
    output = []
    index = 0
    while True:
        added = False
        for name in sorted(shuffled):
            if index < len(shuffled[name]):
                output.append(shuffled[name][index])
                added = True
        if not added:
            return deduplicate(output)
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--software", choices=["AutoRT", "DeepLC"], required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    settings = config["scaling"]
    seed = int(settings["seeds"][args.software])
    pools = load_grouped_pools(Path(config["paths"]["prepared_root"]), "by_species")
    eligible = {name: rows for name, rows in pools.items() if name != "unknown" and len(rows) >= settings["minimum_training_group_size"]}
    root = Path(config["paths"]["scaling_root"]) / "splits" / args.software.lower()
    tasks = []
    for species in settings["species_groups"]:
        candidates = list(pools[species])
        random.Random(stable_seed("species-1k-test", seed, species)).shuffle(candidates)
        test_rows = candidates[:settings["test_size"]]
        if len(test_rows) != settings["test_size"]:
            raise ValueError(f"{species} does not contain {settings['test_size']} peptides")
        nested_pool = balanced_nested_pool({name: rows for name, rows in eligible.items() if name != species}, seed, species)
        nested_pool = [row for row in nested_pool if row_key(row) not in {row_key(item) for item in test_rows}]
        if len(nested_pool) < max(settings["training_sizes"]):
            raise ValueError(f"{species} training pool is smaller than the largest requested training size")
        split_root = root / safe_name(species)
        test_path = split_root / "test.csv"
        write_rows(test_rows, test_path)
        for train_size in settings["training_sizes"]:
            train_path = split_root / f"train_{train_size}.csv"
            write_rows(nested_pool[:train_size], train_path)
            tasks.append({"task_id": f"scaling__{args.software.lower()}__{safe_name(species)}__n{train_size}", "phase": "scaling", "split_type": "by_species", "split_name": species, "round_id": 1, "seed": seed, "train_size": train_size, "train_csv": str(train_path), "test_csv": str(test_path)})
    write_tasks(tasks, Path(config["paths"]["scaling_root"]) / f"scaling_tasks_{args.software.lower()}.tsv")


if __name__ == "__main__":
    main()
