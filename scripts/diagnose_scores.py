import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import load_image

model = tf.keras.models.load_model(ROOT / "artifacts/full/best.keras")

for split in ("train", "validation", "test"):
    data = pd.read_csv(ROOT / "data/splits" / f"{split}.csv")
    sample = data.groupby("label", group_keys=False).sample(
        n=100, random_state=42
    )

    images = []
    for relative_path in sample["image_path"]:
        path = ROOT / relative_path
        images.append(load_image(str(path).encode("utf-8")))

    scores = model.predict(np.stack(images), batch_size=8, verbose=0).ravel()

    print(f"\n{split}")
    for label in ("real", "fake"):
        label_scores = scores[sample["label"].to_numpy() == label]
        print(
            f"  {label}: mean={label_scores.mean():.3f}, "
            f"median={np.median(label_scores):.3f}, "
            f">0.5={(label_scores > 0.5).mean():.1%}"
        )