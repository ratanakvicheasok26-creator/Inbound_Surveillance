"""Real-photo clothing-change test using YuNet + SFace.

Usage:
    python test_clothing_change_photos.py photo_black_shirt.jpg photo_tan_jacket.jpg
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import connect
from face_id import CustomerFaceGallery, FaceRecognizer
from reid import appearance_embedding
from paths import data_dir

DEFAULT_PHOTOS = [
    Path(
        "/home/ratanakvichea/.cursor/projects/home-ratanakvichea-Ibound-survillence-V2/assets/"
        "photo_6165496888979821067_w-0a373527-aa8a-4d77-ab6b-347ae132996f.jpg"
    ),
    Path(
        "/home/ratanakvichea/.cursor/projects/home-ratanakvichea-Ibound-survillence-V2/assets/"
        "photo_6165496888979821066_w-b1124499-ee13-40a7-822e-f4be68e4ca6f.jpg"
    ),
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32).flatten()
    vb = np.asarray(b, dtype=np.float32).flatten()
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-6 or nb <= 1e-6:
        return 0.0
    return float(np.dot(va / na, vb / nb))


def run_clothing_change_test(photo_a: Path, photo_b: Path, threshold: float = 0.60) -> dict:
    img_a = cv2.imread(str(photo_a))
    img_b = cv2.imread(str(photo_b))
    if img_a is None:
        raise FileNotFoundError(f"Could not read {photo_a}")
    if img_b is None:
        raise FileNotFoundError(f"Could not read {photo_b}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        faces_dir = tmp / "faces"
        faces_dir.mkdir()
        conn = connect(tmp / "events.db")
        rec = FaceRecognizer(
            faces_dir=faces_dir,
            models_dir=data_dir() / "models",
            match_threshold=threshold,
            conn=conn,
        )

        emb_a = rec.extract_embedding_from_image(img_a)
        emb_b = rec.extract_embedding_from_image(img_b)
        if emb_a is None:
            raise RuntimeError(f"No face detected in {photo_a.name}")
        if emb_b is None:
            raise RuntimeError(f"No face detected in {photo_b.name}")

        face_sim = _cosine(emb_a, emb_b)
        clothes_sim = _cosine(appearance_embedding(img_a), appearance_embedding(img_b))

        day1, is_new_1, _ = rec.customer_gallery.match_or_enroll(emb_a, now=1_000.0)
        day2, is_new_2, score_2 = rec.customer_gallery.match_or_enroll(emb_b, now=1_000.0 + 86_400.0)

        reloaded = CustomerFaceGallery(threshold=threshold, conn=conn)
        after_restart, is_new_restart, score_restart = reloaded.match_or_enroll(
            emb_b, now=1_000.0 + 172_800.0
        )

        same_person = (not is_new_2) and day2.customer_id == day1.customer_id
        survived_restart = (not is_new_restart) and after_restart.customer_id == day1.customer_id

        result = {
            "photo_a": str(photo_a),
            "photo_b": str(photo_b),
            "face_cosine": face_sim,
            "clothing_cosine": clothes_sim,
            "threshold": threshold,
            "day1_id": day1.customer_id,
            "day1_name": day1.display_name,
            "day2_id": day2.customer_id,
            "day2_name": day2.display_name,
            "day2_is_new": is_new_2,
            "day2_score": score_2,
            "same_person": same_person,
            "restart_id": after_restart.customer_id,
            "restart_is_new": is_new_restart,
            "restart_score": score_restart,
            "survived_restart": survived_restart,
        }
        conn.close()
        return result


def print_result(result: dict) -> None:
    print("=" * 70)
    print(" Clothing-change face test (YuNet detector + SFace embedding)")
    print("=" * 70)
    print(f"  Photo A (black shirt): {result['photo_a']}")
    print(f"  Photo B (tan jacket):  {result['photo_b']}")
    print(f"  Face cosine similarity:     {result['face_cosine']:.3f}  (match if >= {result['threshold']:.2f})")
    print(f"  Clothing/HSV cosine:        {result['clothing_cosine']:.3f}  (body ReID; clothes should differ)")
    print(f"  Day 1 enroll: {result['day1_name']} ({result['day1_id']})")
    print(
        f"  Day 2 match:  {result['day2_name']} ({result['day2_id']}) "
        f"score={result['day2_score']:.3f} new={result['day2_is_new']}"
    )
    print(f"  Same person (face gallery): {'YES' if result['same_person'] else 'NO — treated as different people'}")
    print(
        f"  After SQLite reload:        {'YES same id' if result['survived_restart'] else 'NO — new id'} "
        f"({result['restart_id']}, score={result['restart_score']:.3f})"
    )
    print("=" * 70)


class ClothingChangePhotoTests(unittest.TestCase):
    def test_two_outfit_photos_are_same_customer(self) -> None:
        missing = [p for p in DEFAULT_PHOTOS if not p.is_file()]
        if missing:
            self.skipTest(f"Clothing-change photos not found: {missing}")
        result = run_clothing_change_test(DEFAULT_PHOTOS[0], DEFAULT_PHOTOS[1])
        print_result(result)
        self.assertGreater(result["face_cosine"], 0.20, "Detector/recognizer returned unrelated embeddings")
        self.assertTrue(
            result["same_person"],
            f"Expected same customer across outfits; face cosine={result['face_cosine']:.3f}",
        )
        self.assertTrue(result["survived_restart"], "Face embedding did not survive SQLite reload")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test face match across two outfit photos")
    parser.add_argument("photo_a", nargs="?", default=str(DEFAULT_PHOTOS[0]))
    parser.add_argument("photo_b", nargs="?", default=str(DEFAULT_PHOTOS[1]))
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()
    result = run_clothing_change_test(Path(args.photo_a), Path(args.photo_b), threshold=args.threshold)
    print_result(result)
    return 0 if result["same_person"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("test_"):
        unittest.main()
    else:
        raise SystemExit(main())
