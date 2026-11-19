import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from utilities.synthetic_data import add_occlusion, save_preview


# Samma produkt och dess olika bilder får bara finnas på ena sidan av uppdelningen.
def create_detection_dataset(source, output, background_directory, seed=42):
    cv2.setNumThreads(1)
    source, output = Path(source).resolve(), Path(output).resolve()
    with (source / "annotations.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    products = defaultdict(list)
    for row in rows:
        products[row["product_name"]].append(row)
    rng = random.Random(seed)
    names = sorted(products)
    rng.shuffle(names)
    names = names[:len(names) // 2]
    train_count = int(len(names) * 0.8)
    training_products = set(names[:train_count])
    preview_names = set(names[:2] + names[train_count:train_count + 2])
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    fields = list(rows[0]) + ["product_id", "negative"]
    selected = {"train": [], "val": []}
    preview = []
    backgrounds = sorted(Path(background_directory).glob("*.jpg"))
    if not backgrounds:
        raise ValueError("Bakgrundsbilder saknas.")

    with (output / "annotations.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        # Återanvänd de färdiga bilderna, men skriv nya labels med klass 0 = product.
        for number, name in enumerate(names, start=1):
            split = "train" if name in training_products else "val"
            product_records = []
            for row in products[name]:
                original = source / row["image"]
                image_path = Path("images") / split / original.name
                try:
                    os.link(original, output / image_path)
                except OSError:
                    shutil.copy2(original, output / image_path)
                original_label = source / "labels" / row["split"] / (original.stem + ".txt")
                coordinates = original_label.read_text().split(maxsplit=1)[1]
                (output / "labels" / split / (original.stem + ".txt")).write_text("0 " + coordinates)
                record = dict(row, image=str(image_path), split=split, class_id=0, product_id=row["class_id"], negative=False)
                writer.writerow(record)
                selected[split].append(record)
                product_records.append(record)
            if name in preview_names:
                preview.append(product_records)
            if number % 1000 == 0:
                print(f"{number}/{len(names)} produkter förberedda", flush=True)

        # Tomma händer och tom kylbakgrund ska inte räknas som en produkt.
        negative_preview = []
        negative_counts = {}
        for split_number, split in enumerate(("train", "val")):
            negative_counts[split] = len(selected[split]) // 10
            for number in range(negative_counts[split]):
                sample = rng.choice(selected[split])
                background = backgrounds[number % len(backgrounds)]
                image = cv2.imread(str(background))
                image = cv2.resize(image, (int(sample["width"]), int(sample["height"])), interpolation=cv2.INTER_AREA)
                hand = number % 2 == 1
                if hand:
                    box = tuple(int(sample[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
                    add_occlusion(image, box, rng)
                noise = np.random.default_rng(seed + number + split_number * 100000).normal(0, 1.2, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) * rng.uniform(0.90, 1.08) + rng.uniform(-5, 5) + noise, 0, 255).astype(np.uint8)
                stem = f"empty_{number:05d}"
                image_path = Path("images") / split / (stem + ".jpg")
                if not cv2.imwrite(str(output / image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise OSError(f"Kunde inte spara {image_path}")
                (output / "labels" / split / (stem + ".txt")).write_text("")
                record = {key: "" for key in fields}
                record.update(image=str(image_path), split=split, width=image.shape[1], height=image.shape[0], background_frame=background.stem, variant="empty_hand" if hand else "empty", negative=True)
                writer.writerow(record)
                if number < 3:
                    negative_preview.append(dict(record, product_name="No product"))
            print(f"{split}: {len(selected[split])} produktbilder och {negative_counts[split]} negativa bilder", flush=True)

    settings = {"path": str(output), "train": "images/train", "val": "images/val", "names": ["product"]}
    (output / "dataset.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    (output / "classes.txt").write_text("product\n", encoding="utf-8")
    save_preview(output, preview + [negative_preview])
    print(f"Klart: {train_count} träningsprodukter och {len(names) - train_count} andra produkter för validering. {output / 'dataset.yaml'}", flush=True)
