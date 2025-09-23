#!/usr/bin/env python3
"""
Download a 'no weapon' dataset from COCO 2017 by selecting images that
DO NOT contain blacklisted categories (default: knife), and optionally
DO contain 'person' to match surveillance context.

Outputs a YOLO-style images/labels structure with EMPTY .txt label files
(one per image) to represent "no objects" (i.e., 'no_weapon').

Example:
  python scripts/download_no_weapon_coco.py ^
    --split train2017 ^
    --outdir dataset/no_weapon_coco ^
    --max-images 1200 ^
    --blacklist "knife" ^
    --person-only ^
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple
from zipfile import ZipFile, BadZipFile

import requests
from tqdm import tqdm

COCO_ANN_ZIP_URL_2017 = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_ZIP_NAME = "annotations_trainval2017.zip"
ANN_JSON_TPL = "annotations/instances_{split}.json"  # inside the zip


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, desc: str = "download", chunk_size: int = 1 << 14) -> None:
    if dest.exists():
        # Check if it's a valid zip file
        if dest.suffix.lower() == '.zip':
            try:
                with ZipFile(dest, 'r') as zf:
                    zf.testzip()  # Test zip integrity
                print(f"[INFO] Using cached file: {dest}")
                return
            except (BadZipFile, Exception):
                print(f"[WARN] Cached file {dest} is corrupted, re-downloading...")
                dest.unlink()  # Remove corrupted file
        else:
            return
    
    print(f"[INFO] Downloading {url} -> {dest}")
    
    # Try HTTPS first, fallback to HTTP if SSL fails
    urls_to_try = [url]
    if url.startswith('https://'):
        urls_to_try.append(url.replace('https://', 'http://'))
    
    last_error = None
    for try_url in urls_to_try:
        try:
            # Disable SSL verification for problematic certificates
            verify_ssl = not try_url.startswith('https://images.cocodataset.org')
            with requests.get(try_url, stream=True, timeout=60, verify=verify_ssl) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            break  # Success, exit the retry loop
        except Exception as e:
            last_error = e
            print(f"[WARN] Failed to download from {try_url}: {e}")
            if dest.exists():
                dest.unlink()  # Clean up partial download
            continue
    else:
        # All URLs failed
        raise Exception(f"Failed to download from any URL. Last error: {last_error}")
    
    # Verify the downloaded zip file
    if dest.suffix.lower() == '.zip':
        try:
            with ZipFile(dest, 'r') as zf:
                zf.testzip()
            print(f"[INFO] Successfully downloaded and verified: {dest}")
        except (BadZipFile, Exception) as e:
            print(f"[ERROR] Downloaded file {dest} is not a valid zip: {e}")
            dest.unlink()  # Remove corrupted file
            raise


def extract_json_from_zip(zip_path: Path, internal_json_path: str, out_json_path: Path) -> None:
    if out_json_path.exists():
        return
    try:
        with ZipFile(zip_path, "r") as zf:
            with zf.open(internal_json_path) as jf:
                data = jf.read()
        out_json_path.write_bytes(data)
        print(f"[INFO] Extracted {internal_json_path} successfully")
    except (BadZipFile, Exception) as e:
        print(f"[ERROR] Failed to extract from {zip_path}: {e}")
        print(f"[INFO] Removing corrupted zip file and retrying download...")
        zip_path.unlink()  # Remove corrupted file
        raise


def build_coco_index(instances_json_path: Path) -> Tuple[Dict[int, dict], Dict[int, Set[int]], Dict[str, int]]:
    """
    Returns:
      images_by_id: id -> image dict (includes 'file_name', 'coco_url', etc.)
      img_to_cat_ids: image_id -> set(category_id)
      cat_name_to_id: category name (lower) -> id
    """
    data = json.loads(instances_json_path.read_text(encoding="utf-8"))
    images_by_id = {img["id"]: img for img in data["images"]}
    cat_name_to_id = {c["name"].lower(): c["id"] for c in data["categories"]}
    img_to_cat_ids: Dict[int, Set[int]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        s = img_to_cat_ids.get(img_id)
        if s is None:
            s = set()
            img_to_cat_ids[img_id] = s
        s.add(cat_id)
    return images_by_id, img_to_cat_ids, cat_name_to_id


def filter_images_no_blacklist_and_optional_person(
    images_by_id: Dict[int, dict],
    img_to_cat_ids: Dict[int, Set[int]],
    cat_name_to_id: Dict[str, int],
    blacklist_names: List[str],
    require_person: bool,
) -> List[int]:
    # Resolve category ids
    blacklist_ids = {cat_name_to_id[name.lower()] for name in blacklist_names if name.lower() in cat_name_to_id}
    person_id = cat_name_to_id.get("person", None)

    allowed_ids: List[int] = []
    for img_id, _ in images_by_id.items():
        cats = img_to_cat_ids.get(img_id, set())
        if blacklist_ids & cats:
            continue  # contains a blacklisted category
        if require_person and person_id is not None and person_id not in cats:
            continue  # require person present
        allowed_ids.append(img_id)
    return allowed_ids


def yolo_empty_label_for(stem: str, labels_dir: Path) -> Path:
    return labels_dir / f"{stem}.txt"


def download_image(img_info: dict, out_path: Path, retries: int = 3, backoff: float = 1.5) -> bool:
    url = img_info.get("coco_url") or img_info.get("cocoUrl") or ""
    if not url:
        return False
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return True
        except Exception:
            time.sleep(backoff * (attempt + 1))
    return False


def split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    idxs = list(range(n))
    random.shuffle(idxs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Create a 'no_weapon' dataset from COCO 2017.")
    parser.add_argument("--split", type=str, default="train2017", choices=["train2017", "val2017"],
                        help="COCO split to sample from")
    parser.add_argument("--outdir", type=Path, default=Path("dataset/no_weapon_coco"),
                        help="Output base directory")
    parser.add_argument("--max-images", type=int, default=1000,
                        help="Max number of images to download")
    parser.add_argument("--blacklist", type=str, default="knife",
                        help="Comma-separated category names to exclude (e.g., 'knife,baseball bat')")
    parser.add_argument("--person-only", action="store_true",
                        help="Require 'person' present in the image (recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    args = parser.parse_args()

    random.seed(args.seed)

    # Prepare cache for annotations
    cache_dir = args.outdir / "_annotations_cache"
    ensure_dir(cache_dir)
    zip_path = cache_dir / ANN_ZIP_NAME
    json_inside_zip = ANN_JSON_TPL.format(split=args.split)
    instances_json_path = cache_dir / f"instances_{args.split}.json"

    print(f"[INFO] Downloading COCO annotations zip (once): {COCO_ANN_ZIP_URL_2017}")
    download_file(COCO_ANN_ZIP_URL_2017, zip_path, desc="annotations_zip")

    print(f"[INFO] Extracting {json_inside_zip} -> {instances_json_path}")
    extract_json_from_zip(zip_path, json_inside_zip, instances_json_path)

    print("[INFO] Building COCO indices...")
    images_by_id, img_to_cat_ids, cat_name_to_id = build_coco_index(instances_json_path)

    blacklist_names = [s.strip() for s in args.blacklist.split(",") if s.strip()]
    print(f"[INFO] Blacklist categories: {blacklist_names}")
    allowed_img_ids = filter_images_no_blacklist_and_optional_person(
        images_by_id, img_to_cat_ids, cat_name_to_id, blacklist_names, args.person_only
    )
    print(f"[INFO] Eligible images after filtering: {len(allowed_img_ids)}")

    if len(allowed_img_ids) == 0:
        print("[ERROR] No eligible images found with the given filters. Try relaxing filters or different split.")
        sys.exit(2)

    # Sample
    if args.max_images < len(allowed_img_ids):
        sampled_ids = random.sample(allowed_img_ids, args.max_images)
    else:
        sampled_ids = allowed_img_ids

    # Prepare output folders
    for split_name in ("train", "val", "test"):
        ensure_dir(args.outdir / split_name / "images")
        ensure_dir(args.outdir / split_name / "labels")

    # Split train/val/test
    n = len(sampled_ids)
    train_idx, val_idx, test_idx = split_indices(n, args.train_ratio, args.val_ratio, args.test_ratio)
    idx_to_split = {}
    for i in train_idx:
        idx_to_split[i] = "train"
    for i in val_idx:
        idx_to_split[i] = "val"
    for i in test_idx:
        idx_to_split[i] = "test"

    # Download with progress
    images_list = [images_by_id[iid] for iid in sampled_ids]
    successes = 0
    failures = 0
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, img in enumerate(images_list):
            split_name = idx_to_split[i]
            dest_img = args.outdir / split_name / "images" / img["file_name"]
            # schedule
            futures.append(ex.submit(download_image, img, dest_img))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="download_images"):
            ok = False
            try:
                ok = _.result()
            except Exception:
                ok = False
            if ok:
                successes += 1
            else:
                failures += 1

    # Create empty labels for successfully downloaded images
    # Recompute by checking filesystem
    created_labels = 0
    for split_name in ("train", "val", "test"):
        img_dir = args.outdir / split_name / "images"
        lbl_dir = args.outdir / split_name / "labels"
        for img_path in img_dir.glob("*.*"):
            stem = img_path.stem
            lbl_path = yolo_empty_label_for(stem, lbl_dir)
            if not lbl_path.exists():
                lbl_path.write_text("", encoding="utf-8")
                created_labels += 1

    print(f"[SUMMARY] Downloaded OK: {successes}, failed: {failures}, labels created: {created_labels}")
    print(f"[NEXT] Use this folder as a 'no_weapon' negative set in your unified YOLO data.yaml.")
    print(f"      Example data.yaml entries:")
    print(f"        train: {args.outdir / 'train' / 'images'}")
    print(f"        val:   {args.outdir / 'val' / 'images'}")
    print(f"        test:  {args.outdir / 'test' / 'images'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(130)
