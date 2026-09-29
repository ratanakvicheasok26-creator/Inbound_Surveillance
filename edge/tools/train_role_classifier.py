"""Train and Export YOLO11 Role Classifier for Inbound Surveillance.

Trains a lightweight YOLO11 classification model (yolo11n-cls) to classify
person crops into roles: customer, security_guard, spa_staff, delivery.

Usage:
    python tools/train_role_classifier.py --data dataset/roles --epochs 30 --export
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


def prepare_dataset_splits(data_dir: Path) -> Path:
    """Ensure data is structured into train/ and val/ folders for Ultralytics."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if train_dir.exists() and val_dir.exists():
        return data_dir

    # Split 80/20 if not already split
    print("[Trainer] Preparing train/val splits from dataset...")
    temp_root = data_dir.parent / f"{data_dir.name}_split"
    temp_train = temp_root / "train"
    temp_val = temp_root / "val"
    temp_train.mkdir(parents=True, exist_ok=True)
    temp_val.mkdir(parents=True, exist_ok=True)

    for class_folder in data_dir.iterdir():
        if not class_folder.is_dir() or class_folder.name in ("train", "val"):
            continue
        cname = class_folder.name
        (temp_train / cname).mkdir(parents=True, exist_ok=True)
        (temp_val / cname).mkdir(parents=True, exist_ok=True)

        images = [f for f in class_folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        split_idx = int(len(images) * 0.8)

        for img in images[:split_idx]:
            shutil.copy2(img, temp_train / cname / img.name)
        for img in images[split_idx:]:
            shutil.copy2(img, temp_val / cname / img.name)

    return temp_root


def train_and_export(
    data_dir: str | Path,
    epochs: int = 30,
    imgsz: int = 224,
    batch: int = 16,
    export_models: bool = True,
    output_model_dir: str | Path = "models",
) -> Path:
    data_path = Path(data_dir)
    out_models = Path(output_model_dir)
    out_models.mkdir(parents=True, exist_ok=True)

    split_data = prepare_dataset_splits(data_path)

    print(f"[Trainer] Loading base pretrained yolo11n-cls.pt...")
    model = YOLO("yolo11n-cls.pt")

    print(f"[Trainer] Starting training on {split_data} for {epochs} epochs (imgsz={imgsz})...")
    results = model.train(
        data=str(split_data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/role_cls",
        name="champei_roles",
        exist_ok=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    dest_pt = out_models / "yolo11n-role-cls.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, dest_pt)
        print(f"[Trainer] Saved trained PyTorch model to {dest_pt}")

    if export_models and dest_pt.exists():
        print(f"[Trainer] Exporting to ONNX and OpenVINO...")
        trained_model = YOLO(str(dest_pt))
        try:
            onnx_file = trained_model.export(format="onnx", imgsz=imgsz)
            print(f"[Trainer] Exported ONNX: {onnx_file}")
        except Exception as exc:
            print(f"[Trainer] ONNX export note: {exc}")

        try:
            openvino_file = trained_model.export(format="openvino", imgsz=imgsz)
            print(f"[Trainer] Exported OpenVINO: {openvino_file}")
        except Exception as exc:
            print(f"[Trainer] OpenVINO export note: {exc}")

    print("[Trainer] Training & model export complete!")
    return dest_pt


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 Role Classifier on person crops.")
    parser.add_argument("--data", type=str, default="dataset/roles", help="Path to roles dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size for classification (default 224)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--export", action="store_true", default=True, help="Export to ONNX/OpenVINO after training")
    parser.add_argument("--out", type=str, default="models", help="Output directory for saved weights")
    args = parser.parse_args()

    train_and_export(
        data_dir=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        export_models=args.export,
        output_model_dir=args.out,
    )


if __name__ == "__main__":
    main()
