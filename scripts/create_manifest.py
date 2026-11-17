import csv
from pathlib import Path

project = Path(__file__).resolve().parents[1]
dataset = project / "data" / "raw" / "Dataset"
output = project / "data" / "splits"
output.mkdir(parents=True, exist_ok=True)

for source_split, output_name in [
    ("Train", "train.csv"),
    ("Validation", "validation.csv"),
    ("Test", "test.csv"),
]:
    rows = []

    for folder_label in ("Real", "Fake"):
        folder = dataset / source_split / folder_label
        label = folder_label.lower()

        for image in folder.rglob("*"):
            if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                relative_path = image.relative_to(project).as_posix()
                rows.append({
                    "image_path": relative_path,
                    "label": label,
                    "group_id": relative_path,
                })

    if not rows:
        raise FileNotFoundError(
            f"No images found for {source_split}. Check that the ZIP was extracted "
            f"under {dataset}"
        )

    with (output / output_name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label", "group_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{output_name}: {len(rows)} images")