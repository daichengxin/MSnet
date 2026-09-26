from __future__ import annotations

import argparse
import random
from pathlib import Path

from rt_utils import load_grouped_pools, make_nested_train, read_config, safe_name, stable_seed, write_rows, write_tasks


def eligible_training_names(split_type: str, pools: dict[str, list[dict]], config: dict) -> set[str]:
    if split_type == "by_first_char":
        return {name for name, frame in pools.items() if name != "C"}
    minimum = config["benchmark"]["minimum_training_group_size"]
    excluded = set(config["benchmark"]["excluded_species"])
    return {name for name, frame in pools.items() if name not in excluded and len(frame) >= minimum}


def build_split(split_type: str, split_name: str, pools: dict[str, list[dict]], config: dict, root: Path) -> list[dict]:
    benchmark = config["benchmark"]
    test_size = benchmark["test_size"]
    round_seeds = benchmark["round_seeds"]
    held_out = list(pools[split_name])
    if len(held_out) < test_size * len(round_seeds):
        raise ValueError(f"{split_type}/{split_name} does not contain enough peptides for non-overlapping test rounds")
    random.Random(stable_seed("test", split_type, split_name)).shuffle(held_out)
    training_names = eligible_training_names(split_type, pools, config) - {split_name}
    training_pool = {name: pools[name] for name in training_names}
    tasks = []
    for round_index, seed in enumerate(round_seeds, start=1):
        test_frame = held_out[(round_index - 1) * test_size:round_index * test_size]
        split_root = root / split_type / safe_name(split_name) / f"round_{round_index}"
        test_path = split_root / "test.csv"
        write_rows(test_frame, test_path)
        train_seed = stable_seed("train", split_type, split_name, round_index, seed)
        nested = make_nested_train(training_pool, {row["key"] for row in test_frame}, train_seed, benchmark["training_sizes"])
        for train_size, train_rows in nested.items():
            train_path = split_root / f"train_{train_size}.csv"
            write_rows(train_rows, train_path)
            tasks.append({
                "task_id": f"benchmark__{split_type}__{safe_name(split_name)}__r{round_index}__n{train_size}",
                "phase": "benchmark", "split_type": split_type, "split_name": split_name,
                "round_id": round_index, "seed": seed, "train_size": train_size,
                "train_csv": str(train_path), "test_csv": str(test_path)
            })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    root = Path(config["paths"]["benchmark_root"]) / "splits"
    tasks = []
    selected = (("by_first_char", config["benchmark"]["first_character_groups"]), ("by_species", config["benchmark"]["species_groups"]))
    for split_type, groups in selected:
        pools = load_grouped_pools(Path(config["paths"]["prepared_root"]), split_type)
        for group in groups:
            tasks.extend(build_split(split_type, group, pools, config, root))
    write_tasks(tasks, Path(config["paths"]["benchmark_root"]) / "benchmark_tasks.tsv")


if __name__ == "__main__":
    main()
