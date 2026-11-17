import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SIZE = (224, 224)


def load_image(path_bytes):
    path = os.fsdecode(path_bytes.item() if isinstance(path_bytes, np.ndarray) else path_bytes)
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces):
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        px, py = int(w * 0.15), int(h * 0.15)
        rgb = rgb[max(0, y-py):min(rgb.shape[0], y+h+py),
                  max(0, x-px):min(rgb.shape[1], x+w+px)]

    return cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def load_split(csv_path, batch_size, training=False, seed=42):
    data = pd.read_csv(csv_path)
    required = {"image_path", "label", "group_id"}
    if not required.issubset(data.columns):
        raise ValueError(f"{csv_path} must contain image_path,label,group_id")

    paths = []
    for value in data["image_path"]:
        path = Path(value)
        paths.append(str(path if path.is_absolute() else ROOT / path))

    labels = data["label"].str.lower().map({"real": 0, "fake": 1})
    if labels.isna().any():
        raise ValueError(f"Unexpected label in {csv_path}")
    labels = labels.to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True)

    def prepare(path, label):
        image = tf.numpy_function(load_image, [path], tf.float32)
        image.set_shape((SIZE[0], SIZE[1], 3))
        return image, label

    ds = ds.map(prepare, num_parallel_calls=1)
    ds = ds.batch(batch_size).prefetch(1)
    return ds, labels


def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs
    )
    backbone.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    train_ds, _ = load_split(args.splits / "train.csv", args.batch_size, True, args.seed)
    val_ds, _ = load_split(args.splits / "validation.csv", args.batch_size)
    test_ds, test_y = load_split(args.splits / "test.csv", args.batch_size)

    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output / "best.keras", monitor="val_auc", mode="max", save_best_only=True
        ),
    ]
    model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=callbacks,
    shuffle=False,
    )
    model = tf.keras.models.load_model(args.output / "best.keras")

    scores = model.predict(test_ds, verbose=0).ravel()
    predicted = (scores >= 0.5).astype(int)
    metrics = {
        "threshold": 0.5,
        "accuracy": float(accuracy_score(test_y, predicted)),
        "roc_auc": float(roc_auc_score(test_y, scores)),
        "average_precision": float(average_precision_score(test_y, scores)),
        "confusion_matrix_real_fake": confusion_matrix(test_y, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            test_y, predicted, labels=[0, 1], target_names=["real", "fake"],
            output_dict=True, zero_division=0
        ),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()