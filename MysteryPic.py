#!/usr/bin/env python3
"""
Mystery Pic Search
------------------

Author: @willnjohnson  
Description:  
This script scans Neopets banners to help identify the current Mystery Pic by comparing 
a provided image (PNG or GIF) against stored banner images using SSIM (Structural Similarity Index).

Features:
- Automatically converts GIFs to PNG for scanning.
- Resizes input images to 10x10 for consistent matching.
- Uses SSIM to compare each region of banner images.
- Early exit option if a match is found above threshold.
- Parallel processing using multiple CPU cores.

Usage:
  python MysteryPic.py -e       # Stops on first strong match
  python MysteryPic.py --help   # Prints usage info

Folder Requirements:
- `MysteryPic/` must contain exactly one image file (GIF or PNG).
- `MysteryPic/banners/` must contain PNG banner images (e.g. shopkeeper banners) to scan. Download them manually (e.g. https://www.drsloth.com/search/?category=18).

"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import concurrent.futures

def convert_gif_to_png(gif_path, png_path):
    try:
        with Image.open(gif_path) as im:
            im = im.convert("RGBA")  # convert to RGBA to preserve transparency if any
            im = im.resize((10, 10), Image.Resampling.LANCZOS)
            im.save(png_path, format="PNG")
        print(f"Converted GIF to PNG: {png_path}")
        return True
    except Exception as e:
        print(f"Error converting GIF to PNG: {e}")
        return False

def calculate_ssim_similarity(image1_np, image2_np):
    try:
        image1_gray = cv2.cvtColor(image1_np, cv2.COLOR_BGR2GRAY) if len(image1_np.shape) == 3 else image1_np
        image2_gray = cv2.cvtColor(image2_np, cv2.COLOR_BGR2GRAY) if len(image2_np.shape) == 3 else image2_np
        score, _ = ssim(image1_gray.astype("float64"), image2_gray.astype("float64"), full=True, data_range=255)
        return (score + 1) / 2
    except Exception as e:
        print(f"\nError in calculate_ssim_similarity: {e}")
        return 0.0

def scan_single_banner(template_np, banner_image_path, threshold=0.95):
    try:
        target = cv2.imread(banner_image_path)
        if target is None:
            return None

        template_h, template_w, _ = template_np.shape
        target_h, target_w, _ = target.shape

        if target_h < template_h or target_w < template_w:
            return None

        best_prob = -1.0
        best_loc = None
        template_gray = cv2.cvtColor(template_np, cv2.COLOR_BGR2GRAY).astype("float64")

        for y in range(0, target_h - template_h + 1):
            for x in range(0, target_w - template_w + 1):
                block = target[y:y + template_h, x:x + template_w]
                block_gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY).astype("float64")
                score, _ = ssim(template_gray, block_gray, full=True, data_range=255)
                similarity = (score + 1) / 2
                if similarity > best_prob:
                    best_prob = similarity
                    best_loc = (x, y)

        if best_prob >= threshold:
            return {
                "filename": os.path.basename(banner_image_path),
                "probability": best_prob,
                "location": best_loc
            }
        return None
    except Exception as e:
        print(f"\nError processing banner '{banner_image_path}': {e}")
        return None

def print_help():
    print("""
Mystery Pic Banner Scanner - Usage Guide

This script helps you find where the current Mystery Pic is hiding among banners.

Optional Flags:
  -e, --early     Stops scanning as soon as a match above the threshold is found.
  -h, --help      Displays this help message.

Requirements:
  Place exactly one image file (PNG or GIF) in the 'MysteryPic/' folder.
  If a GIF is present, it will be automatically converted to a 10x10 PNG for matching.
  Place PNG images of banners inside the 'MysteryPic/banners/' folder.

Examples:
  python MysteryPic.py -e
  python MysteryPic.py --help

""")

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-e", "--early", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    args = parser.parse_args()

    if args.help:
        print_help()
        return

    script_dir = os.path.dirname(__file__)
    mysterypic_dir = os.path.join(script_dir, "MysteryPic")
    banners_dir = os.path.join(mysterypic_dir, "banners")

    if not os.path.exists(mysterypic_dir):
        print(f"Error: MysteryPic directory '{mysterypic_dir}' does not exist. Please create it and add your Mystery Pic image.")
        return

    # Find image files (.png or .gif) in MysteryPic/ (non-recursive)
    image_files = [
        f for f in os.listdir(mysterypic_dir)
        if os.path.isfile(os.path.join(mysterypic_dir, f)) and f.lower().endswith(('.png', '.gif'))
    ]

    if len(image_files) == 0:
        print(f"No PNG or GIF image found in '{mysterypic_dir}'. Please add exactly one Mystery Pic image.")
        return
    elif len(image_files) > 1:
        print(f"Multiple image files found in '{mysterypic_dir}':")
        for f in image_files:
            print(f"  - {f}")
        print("Please keep only one Mystery Pic image file in this folder.")
        return

    mystery_pic_file = image_files[0]
    mystery_pic_path = os.path.join(mysterypic_dir, mystery_pic_file)

    # If GIF, convert to PNG and replace mystery_pic_path with new PNG path
    if mystery_pic_file.lower().endswith(".gif"):
        converted_png_path = os.path.join(mysterypic_dir, "mysterypic_converted.png")
        success = convert_gif_to_png(mystery_pic_path, converted_png_path)
        if not success:
            return
        mystery_pic_path = converted_png_path

    # Read and resize template
    template_img = cv2.imread(mystery_pic_path)
    if template_img is None:
        print(f"Error: Unable to read Mystery Pic image '{mystery_pic_path}'.")
        return

    scaled_template = cv2.resize(template_img, (10, 10), interpolation=cv2.INTER_AREA)
    print(f"Template loaded and resized: {scaled_template.shape[1]}x{scaled_template.shape[0]}")

    # Check banners directory
    if not os.path.exists(banners_dir):
        print(f"Error: Banners directory '{banners_dir}' not found. Please create it and add PNG banner images.")
        return

    banner_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(banners_dir)
        for f in files if f.lower().endswith(".png")
    ]

    if not banner_paths:
        print("No PNG banner images found in the banners folder.")
        return

    print(f"Scanning {len(banner_paths)} banners...")
    start_time = time.time()
    all_matches = []
    early_stop = args.early
    max_workers = os.cpu_count() or 4

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(scan_single_banner, scaled_template, path, 0.95): path
            for path in banner_paths
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scanning"):
            path = futures[future]
            tqdm.write(f"  Done: {os.path.basename(path)}")
            try:
                result = future.result()
                if result:
                    print(f"    Match Found: {result['filename']} @ {result['location']} | Probability: {result['probability']:.4f}")
                    all_matches.append(result)
                    if early_stop:
                        executor.shutdown(cancel_futures=True)
                        break
            except Exception as e:
                print(f"Error scanning {path}: {e}")

    if all_matches:
        print("\n--- High Probability Matches ---")
        all_matches.sort(key=lambda m: m['probability'], reverse=True)
        for m in all_matches:
            print(f"{m['filename']}: {m['probability']:.4f} at {m['location']}")
    else:
        print("No matches found above threshold.")

    print(f"\nFinished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
