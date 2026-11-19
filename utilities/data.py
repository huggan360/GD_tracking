import csv
import random
from collections import defaultdict
from pathlib import Path

import yaml


# Välj halva datasetet men behåll träningsbilder för varje produktnamn.
def create_demo_dataset(source, output, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    with (source / "annotations.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    products = defaultdict(list)
    validation = []
    for row in rows:
        if row["split"] == "train":
            products[row["class_id"]].append(row)
        elif row["split"] == "val":
            validation.append(row)

    rng = random.Random(seed)
    classes = list(products)
    rng.shuffle(classes)
    training = []
    for index, class_id in enumerate(classes):
        images = products[class_id]
        count = max(1, len(images) // 2 + (len(images) % 2 if index % 2 == 0 else 0))
        training.extend(rng.sample(images, count))
    validation = rng.sample(validation, len(validation) // 2)

    output.mkdir(parents=True, exist_ok=True)
    for split, selected in (("train", training), ("val", validation)):
        (output / f"{split}.txt").write_text("\n".join(str(source / row["image"]) for row in selected) + "\n", encoding="utf-8")
    config = yaml.safe_load((source / "dataset.yaml").read_text())
    config.update(path=str(output), train="train.txt", val="val.txt")
    (output / "dataset.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Demo: {len(training)} träningsbilder, {len(validation)} valideringsbilder, {len(classes)} produktnamn.")
