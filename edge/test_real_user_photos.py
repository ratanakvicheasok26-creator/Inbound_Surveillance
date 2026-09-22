"""Test Face Recognition & Re-ID on User's Actual Uploaded Photos with Different Clothes.

Photo 1: Black T-shirt (Parking/Garage)
Photo 2: Brown Jacket + White shirt + Lanyard (Event Crowd)
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from face_id import FaceRecognizer, CustomerFaceGallery
from reid import appearance_embedding

UPLOAD_DIR = Path("/home/ratanakvichea/.gemini/antigravity-ide/brain/b913444a-fcc5-4b94-a80d-bdd23945873a/.user_uploaded")
IMG1_PATH = UPLOAD_DIR / "media_1790055429425.jpg"
IMG2_PATH = UPLOAD_DIR / "media_1790055429461.jpg"


def test_user_photos():
    print("=" * 70)
    print(" 📸 TESTING USER'S REAL PHOTOS WITH DIFFERENT CLOTHES")
    print("=" * 70)

    if not IMG1_PATH.is_file() or not IMG2_PATH.is_file():
        print(f"[test_real_user_photos] Test images not found ({IMG1_PATH}, {IMG2_PATH}). Skipping standalone test.")
        return

    img1 = cv2.imread(str(IMG1_PATH))
    img2 = cv2.imread(str(IMG2_PATH))

    assert img1 is not None, f"Could not load {IMG1_PATH}"
    assert img2 is not None, f"Could not load {IMG2_PATH}"

    print(f"[*] Photo 1 Loaded: {img1.shape[1]}x{img1.shape[0]} (Black T-Shirt)")
    print(f"[*] Photo 2 Loaded: {img2.shape[1]}x{img2.shape[0]} (Brown Jacket + White Shirt)")

    # Initialize Face Recognizer
    face_rec = FaceRecognizer()

    # Extract Face Embeddings
    emb1 = face_rec.extract_embedding_from_image(img1)
    emb2 = face_rec.extract_embedding_from_image(img2)

    print("\n" + "-" * 70)
    print(" 🔍 STEP 1: FACE DETECTION & BIOMETRIC EXTRACTION")
    print("-" * 70)

    if emb1 is None:
        print(" [!] Failed to detect face in Photo 1")
    else:
        print(f" [✔] Photo 1 Face Detected! Embedding norm: {np.linalg.norm(emb1):.3f} (128 dimensions)")

    if emb2 is None:
        print(" [!] Failed to detect face in Photo 2")
    else:
        print(f" [✔] Photo 2 Face Detected! Embedding norm: {np.linalg.norm(emb2):.3f} (128 dimensions)")

    # Compare Clothing / Appearance Embeddings
    print("\n" + "-" * 70)
    print(" 👕 STEP 2: CLOTHING / BODY APPEARANCE COMPARISON (OLD SYSTEM)")
    print("-" * 70)
    body1 = appearance_embedding(img1)
    body2 = appearance_embedding(img2)
    clothing_sim = float(np.dot(body1, body2))
    print(f" Clothing Cosine Similarity: {clothing_sim:.3f}")
    print(f" -> Old Clothing Re-ID Verdict: {'DIFFERENT PERSON (Failed because of clothes)' if clothing_sim < 0.60 else 'SAME'}")

    # Compare Face Biometric Embeddings
    print("\n" + "-" * 70)
    print(" 👤 STEP 3: BIOMETRIC FACE RECOGNITION COMPARISON (NEW SYSTEM)")
    print("-" * 70)
    if emb1 is not None and emb2 is not None:
        cosine_mode = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)
        face_match_score = float(face_rec.recognizer.match(emb1, emb2, cosine_mode))
        
        # Raw dot product between normalized vectors
        norm_emb1 = emb1 / np.linalg.norm(emb1)
        norm_emb2 = emb2 / np.linalg.norm(emb2)
        dot_sim = float(np.dot(norm_emb1.flatten(), norm_emb2.flatten()))

        print(f" OpenCV SFace Match Score: {face_match_score:.4f}")
        print(f" Normalized Vector Dot Similarity: {dot_sim:.4f}")
        print(f" Recognition Threshold: {face_rec.match_threshold:.2f}")

        # Test with CustomerFaceGallery
        gallery = CustomerFaceGallery(threshold=face_rec.match_threshold)
        
        # Day 1: User arrives in Black T-shirt
        prof1, is_new1, _ = gallery.match_or_enroll(emb1, now=1000.0)
        print(f"\n [Day 1 Visit - Black T-shirt]:")
        print(f"   -> System Action: Enrolled as '{prof1.display_name}' (ID: {prof1.customer_id})")
        print(f"   -> Visit Count: {prof1.visit_count}")

        # Day 2: Same user arrives in Brown Jacket + White Shirt
        prof2, is_new2, match_score2 = gallery.match_or_enroll(emb2, now=1000.0 + 86400.0)
        print(f"\n [Day 2 Visit - Brown Jacket]:")
        print(f"   -> System Action: {'RECOGNIZED RETURNING CUSTOMER' if not is_new2 else 'FAILED AS NEW VISITOR'}")
        print(f"   -> Matched Profile: '{prof2.display_name}' (ID: {prof2.customer_id})")
        print(f"   -> Match Confidence: {match_score2:.4f}")
        print(f"   -> Lifetime Visit Count: {prof2.visit_count}")

        print("\n" + "=" * 70)
        if not is_new2 and prof2.customer_id == prof1.customer_id:
            print(" 🎉 CONCLUSION: SUCCESS! The system successfully recognized you as the")
            print(" SAME PERSON across both photos despite having completely different clothes!")
        else:
            print(f" ⚠️ RESULT: Match score was {face_match_score:.4f} (Threshold: {face_rec.match_threshold:.2f})")
        print("=" * 70)


if __name__ == "__main__":
    test_user_photos()
