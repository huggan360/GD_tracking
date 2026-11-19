import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Läs epokernas loss från Ultralytics och rita träning och validering i samma figur.
def plot_loss(csv_file, output):
    with open(csv_file, newline="", encoding="utf-8") as file:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(file)]
    epochs = [int(row["epoch"]) for row in rows]
    train = [sum(float(row[f"train/{name}_loss"]) for name in ("box", "cls", "dfl")) for row in rows]
    validation = [sum(float(row[f"val/{name}_loss"]) for name in ("box", "cls", "dfl")) for row in rows]

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(epochs, train, marker="o", markersize=3, label="Train")
    axes.plot(epochs, validation, marker="o", markersize=3, label="Validation")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss (box + cls + dfl)")
    axes.legend()
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
