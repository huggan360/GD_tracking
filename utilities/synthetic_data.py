import csv
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps


# Bakgrunderna följer med repot och är redan beskurna till YOLO:s arbetsområde.
def read_backgrounds(directory, output, image_height):
    paths = sorted(Path(directory).glob("*.jpg"))
    if not paths:
        raise ValueError(f"Inga bakgrundsbilder finns i {directory}.")
    backgrounds = []
    (output / "backgrounds").mkdir(parents=True)
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Kunde inte läsa bakgrunden {path}.")
        if frame.shape[0] != image_height:
            frame = cv2.resize(frame, (round(frame.shape[1] * image_height / frame.shape[0]), image_height), interpolation=cv2.INTER_AREA)
        shutil.copy2(path, output / "backgrounds" / path.name)
        backgrounds.append((path.stem, frame))
    return backgrounds


# Ta bara bort nästan vitt som hänger ihop med bildkanten.
def cut_out_product(path):
    with Image.open(path) as source:
        image = np.array(ImageOps.exif_transpose(source).convert("RGBA"))
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    scale = min(1, 512 / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    color = image[:, :, :3]
    white = ((color.min(axis=2) > 240) & (color.max(axis=2) - color.min(axis=2) < 20)).astype(np.uint8)
    _, regions = cv2.connectedComponents(white)
    edge_regions = np.unique(np.concatenate((regions[0], regions[-1], regions[:, 0], regions[:, -1])))
    background = (white != 0) & np.isin(regions, edge_regions)
    alpha = image[:, :, 3].astype(np.float32) / 255
    masked_alpha = alpha * ~background
    if np.count_nonzero(masked_alpha) > 0.02 * masked_alpha.size:
        alpha = cv2.GaussianBlur(masked_alpha, (3, 3), 0.5)
    y, x = np.where(alpha > 0.05)
    if len(x) == 0:
        raise ValueError(f"Produkten har inga synliga pixlar: {path}")
    return color[y.min():y.max() + 1, x.min():x.max() + 1], alpha[y.min():y.max() + 1, x.min():x.max() + 1]


# Rotation och perspektiv ändrar produkten, inte kylbakgrunden.
def transform_product(color, alpha, variant, rng, background_shape):
    color = color.astype(np.float32)
    if variant > 0:
        color = np.clip(color * rng.uniform(0.65, 1.20) * np.array([rng.uniform(0.9, 1.1) for _ in range(3)]), 0, 255)
    layer = np.dstack((color * alpha[:, :, None], alpha)).astype(np.float32)
    height, width = alpha.shape
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    destination = source.copy()
    if variant in (2, 3, 5):
        for point in destination:
            point += [rng.uniform(-0.14, 0.14) * width, rng.uniform(-0.14, 0.14) * height]
    angle = rng.uniform(-30, 30) if variant > 0 else rng.uniform(-5, 5)
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    destination = cv2.transform(destination[None], rotation)[0]
    destination -= destination.min(axis=0) - 8
    size = np.ceil(destination.max(axis=0) + 9).astype(int)
    matrix = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
    layer = cv2.warpPerspective(layer, matrix, tuple(int(value) for value in size))
    background_height, background_width = background_shape[:2]
    # Halva bredden och höjden jämfört med det första datasetet.
    scale = 0.5 * min(rng.uniform(0.25, 0.85) * background_width / layer.shape[1], rng.uniform(0.18, 0.50) * background_height / layer.shape[0])
    layer = cv2.resize(layer, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if variant in (4, 5):
        length = rng.choice([3, 5, 7])
        kernel = np.zeros((length, length), np.float32)
        cv2.line(kernel, (0, rng.randrange(length)), (length - 1, rng.randrange(length)), 1, 1)
        layer = cv2.filter2D(layer, -1, kernel / kernel.sum())
    return layer


# Armen kommer alltid från vänster. Fingrar, skuggning och lite oskärpa mjukar upp formen.
def add_occlusion(image, box, rng):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    radius = max(3, round(min(width, height) * rng.uniform(0.18, 0.26)))
    center = np.float32([left + width * rng.uniform(0.18, 0.32), top + height * rng.uniform(0.55, 0.80)])
    start = np.float32([-2 * radius, center[1] + rng.uniform(-0.25, 0.25) * height])
    direction = center - start
    normal = np.float32([-direction[1], direction[0]]) / np.linalg.norm(direction)
    wrist = center - np.float32([radius * 0.5, 0])
    outline = np.array([start - normal * radius, wrist - normal * radius * 0.5, wrist + normal * radius * 0.5, start + normal * radius]).astype(np.int32)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, outline, 255, cv2.LINE_AA)
    palm = tuple(int(value) for value in center)
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    cv2.ellipse(mask, palm, (radius, max(2, round(radius * 0.65))), angle, 0, 360, 255, -1, cv2.LINE_AA)
    for finger in (-0.5, 0, 0.5):
        base = (round(center[0]), round(center[1] + finger * radius))
        tip = (round(center[0] + radius * rng.uniform(1.2, 1.7)), round(center[1] + finger * radius - radius * 0.2))
        cv2.line(mask, base, tip, 255, max(2, round(radius * 0.4)), cv2.LINE_AA)
    cv2.line(mask, palm, (round(center[0] + radius * 0.7), round(center[1] + radius)), 255, max(2, radius // 2), cv2.LINE_AA)

    # Hudtonerna varierar tillsammans så att armen inte får slumpmässigt gröna eller gula färger.
    skin = np.float32(rng.choice([(95, 135, 180), (125, 165, 205), (145, 185, 225), (65, 95, 140), (80, 120, 170)]))
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    distance = np.abs((yy - start[1]) * direction[0] - (xx - start[0]) * direction[1]) / np.linalg.norm(direction)
    shading = 1.06 - 0.22 * np.clip(distance / radius, 0, 1)
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, 1.5, mask.shape).astype(np.float32)
    arm = np.clip(skin * shading[:, :, None] + noise[:, :, None], 0, 255)
    arm = cv2.GaussianBlur(arm, (5, 5), 1.0)
    opacity = cv2.GaussianBlur(mask.astype(np.float32) / 255, (7, 7), 1.1)[:, :, None]
    image[:] = np.clip(arm * opacity + image * (1 - opacity), 0, 255).astype(np.uint8)


# Bilden och dess YOLO-ruta skapas tillsammans, så att annoteringen följer alla transformationer.
def make_image(color, alpha, background, variant, rng):
    image = background.copy()
    layer = transform_product(color, alpha, variant, rng, image.shape)
    height, width = layer.shape[:2]
    x = rng.randint(2, image.shape[1] - width - 2)
    y = rng.randint(round(image.shape[0] * 0.05), max(round(image.shape[0] * 0.05), image.shape[0] - height - 2))
    opacity = np.clip(layer[:, :, 3:4], 0, 1)
    area = image[y:y + height, x:x + width]
    area[:] = np.clip(layer[:, :, :3] + area * (1 - opacity), 0, 255).astype(np.uint8)
    py, px = np.where(opacity[:, :, 0] > 0.05)
    box = (x + int(px.min()), y + int(py.min()), x + int(px.max()) + 1, y + int(py.max()) + 1)
    if variant in (3, 5):
        add_occlusion(image, box, rng)
    if variant > 0:
        image = np.clip(image.astype(np.float32) * rng.uniform(0.90, 1.08) + rng.uniform(-5, 5), 0, 255).astype(np.uint8)
    return image, box


# Sex kolumner visar de sex varianterna av några produkter med sina annoteringar.
def save_preview(output, records):
    canvas = Image.new("RGB", (6 * 220, len(records) * 300), "white")
    draw = ImageDraw.Draw(canvas)
    for row_number, variants in enumerate(records):
        for column, record in enumerate(variants):
            image = Image.open(output / record["image"]).convert("RGB")
            if record["xmin"] != "":
                box = tuple(int(record[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
                ImageDraw.Draw(image).rectangle(box, outline="#00ff40", width=3)
            image.thumbnail((216, 260))
            canvas.paste(image, (column * 220 + (220 - image.width) // 2, row_number * 300 + 36))
            draw.text((column * 220 + 4, row_number * 300 + 3), record["product_name"][:28], fill="black")
            draw.text((column * 220 + 4, row_number * 300 + 18), f"{record['split']} / {record['variant']}", fill="black")
    canvas.save(output / "preview.jpg", quality=92)


# Alla klasser får fem träningsbilder och en ny syntetisk valideringsbild.
def generate_dataset(source, background_directory, output, variants, image_height, workers, seed):
    cv2.setNumThreads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    products = defaultdict(list)
    with (source / "annotations.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            products[row["product_name"].strip()].append(row)
    names = sorted(products)
    backgrounds = read_backgrounds(background_directory, output, image_height)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    settings = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": names}
    (output / "dataset.yaml").write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    variant_names = ["clean", "rotated", "perspective", "occluded", "motion_blur", "mixed"]

    # En produkts olika källbilder används i turordning om vi har fler än en bild.
    def generate_product(class_id):
        rng = random.Random(seed + class_id)
        name = names[class_id]
        rows = products[name]
        photos = [cut_out_product(source / row["image"]) for row in rows]
        records = []
        for variant in range(variants):
            split = "val" if variant == variants - 1 else "train"
            frame_number, background = backgrounds[(class_id + variant) % len(backgrounds)]
            color, alpha = photos[variant % len(photos)]
            image, box = make_image(color, alpha, background, variant, rng)
            stem = f"{class_id:05d}_{variant}"
            image_path = Path("images") / split / f"{stem}.jpg"
            if not cv2.imwrite(str(output / image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError(f"Kunde inte spara {image_path}")
            left, top, right, bottom = box
            height, width = image.shape[:2]
            annotation = f"{class_id} {(left + right) / (2 * width):.6f} {(top + bottom) / (2 * height):.6f} {(right - left) / width:.6f} {(bottom - top) / height:.6f}\n"
            (output / "labels" / split / f"{stem}.txt").write_text(annotation)
            records.append({"image": str(image_path), "product_name": name, "class_id": class_id, "source_image": rows[variant % len(rows)]["image"], "background_frame": frame_number, "variant": variant_names[variant], "xmin": left, "ymin": top, "xmax": right, "ymax": bottom, "width": width, "height": height, "split": split})
        return records

    # Gör översikten först. Dessa bilder ingår också i det färdiga datasetet.
    preview_ids = set(int(value) for value in np.linspace(0, len(names) - 1, 6))
    preview = [generate_product(class_id) for class_id in sorted(preview_ids)]
    save_preview(output, preview)
    print(f"Översikt: {output / 'preview.jpg'}", flush=True)
    with (output / "annotations.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(preview[0][0]))
        writer.writeheader()
        for records in preview:
            writer.writerows(records)
        remaining = [class_id for class_id in range(len(names)) if class_id not in preview_ids]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for completed, records in enumerate(pool.map(generate_product, remaining), start=len(preview) + 1):
                writer.writerows(records)
                if completed % 100 == 0:
                    print(f"{completed}/{len(names)} produkter, {completed * variants} bilder", flush=True)
    print(f"Klart: {len(names)} produktnamn, {len(names) * variants} bilder. Dataset: {output / 'dataset.yaml'}", flush=True)
