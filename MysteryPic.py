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
- Parallel processing using multiple CPU cores and multiple threads.
- Smart 2-pass approach for time improvement
  * Pass 1: Coarse search with step=3 (fast coverage)
  * Pass 2: Fine search around promising areas (step=1)
  
Usage:
  python MysteryPic.py -f       # Enable fast mode
  python MysteryPic.py -e       # Stops on first strong match
  python MysteryPic.py -t 32    # Utilize 32 workers
  python MysteryPic.py --help   # Prints usage info

  Recommended command: python MysteryPic.py -e -f

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
import threading
from functools import partial
import multiprocessing as mp

def convert_gif_to_png(gif_path, png_path):
    try:
        with Image.open(gif_path) as im:
            im = im.convert("RGBA")
            im = im.resize((10, 10), Image.Resampling.LANCZOS)
            im.save(png_path, format="PNG")
        print(f"Converted GIF to PNG: {png_path}")
        return True
    except Exception as e:
        print(f"Error converting GIF to PNG: {e}")
        return False

def calculate_ssim_vectorized(template_gray, target_blocks):
    """Vectorized SSIM calculation for multiple blocks at once"""
    similarities = []
    for block in target_blocks:
        try:
            score, _ = ssim(template_gray, block, full=True, data_range=255)
            similarities.append((score + 1) / 2)
        except:
            similarities.append(0.0)
    return similarities

def scan_banner_optimized(args):
    """Optimized banner scanning - balanced speed and accuracy"""
    template_gray, banner_path, threshold, chunk_size = args
    
    try:
        target = cv2.imread(banner_path)
        if target is None:
            return None

        template_h, template_w = template_gray.shape
        target_h, target_w = target.shape[:2]

        if target_h < template_h or target_w < template_w:
            return None

        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype("float64")
        
        best_prob = -1.0
        best_loc = None
        
        # For 10x10 templates, use step=1 to ensure we don't miss exact matches
        # But process in efficient chunks
        positions = [(x, y) for y in range(0, target_h - template_h + 1)
                     for x in range(0, target_w - template_w + 1)]
        
        # Process in chunks for better memory usage
        for i in range(0, len(positions), chunk_size):
            chunk_positions = positions[i:i + chunk_size]
            
            # Process each position in the chunk
            for x, y in chunk_positions:
                block = target_gray[y:y + template_h, x:x + template_w]
                
                try:
                    score, _ = ssim(template_gray, block, full=True, data_range=255)
                    similarity = (score + 1) / 2
                    
                    if similarity > best_prob:
                        best_prob = similarity
                        best_loc = (x, y)
                        
                        # Early exit if we found a very good match
                        if similarity > 0.99:
                            break
                except:
                    continue
            
            # Early exit for very good matches
            if best_prob > 0.99:
                break

        if best_prob >= threshold:
            return {
                "filename": os.path.basename(banner_path),
                "probability": best_prob,
                "location": best_loc
            }
        return None
        
    except Exception as e:
        print(f"\nError processing banner '{banner_path}': {e}")
        return None

def scan_banner_ultra_fast(args):
    """Ultra-fast scanning with optimized but accurate search"""
    template_gray, banner_path, threshold = args
    
    try:
        target = cv2.imread(banner_path)
        if target is None:
            return None

        template_h, template_w = template_gray.shape
        target_h, target_w = target.shape[:2]

        if target_h < template_h or target_w < template_w:
            return None

        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype("float64")
        
        best_prob = -1.0
        best_loc = None
        
        # For 10x10 templates, use smaller steps to ensure we don't miss matches
        # Two-pass approach: coarse then fine
        
        # Pass 1: Coarse search with step=3 (covers most area quickly)
        coarse_candidates = []
        for y in range(0, target_h - template_h + 1, 3):
            for x in range(0, target_w - template_w + 1, 3):
                block = target_gray[y:y + template_h, x:x + template_w]
                
                try:
                    score, _ = ssim(template_gray, block, full=True, data_range=255)
                    similarity = (score + 1) / 2
                    
                    if similarity > best_prob:
                        best_prob = similarity
                        best_loc = (x, y)
                    
                    # Collect promising areas for fine search
                    if similarity > 0.8:  # Lower threshold for candidates
                        coarse_candidates.append((x, y, similarity))
                        
                except:
                    continue
        
        # Pass 2: Fine search around promising areas
        for cx, cy, _ in coarse_candidates:
            # Search 6x6 area around each candidate (3 pixels in each direction)
            for dy in range(max(0, cy-3), min(target_h - template_h + 1, cy+4)):
                for dx in range(max(0, cx-3), min(target_w - template_w + 1, cx+4)):
                    # Skip if we already checked this exact position
                    if (dx - cx) % 3 == 0 and (dy - cy) % 3 == 0:
                        continue
                        
                    block = target_gray[dy:dy + template_h, dx:dx + template_w]
                    
                    try:
                        score, _ = ssim(template_gray, block, full=True, data_range=255)
                        similarity = (score + 1) / 2
                        
                        if similarity > best_prob:
                            best_prob = similarity
                            best_loc = (dx, dy)
                            
                    except:
                        continue
        
        # If no good candidates found, do a more thorough search with step=2
        if best_prob < 0.9 and not coarse_candidates:
            for y in range(0, target_h - template_h + 1, 2):
                for x in range(0, target_w - template_w + 1, 2):
                    block = target_gray[y:y + template_h, x:x + template_w]
                    
                    try:
                        score, _ = ssim(template_gray, block, full=True, data_range=255)
                        similarity = (score + 1) / 2
                        
                        if similarity > best_prob:
                            best_prob = similarity
                            best_loc = (x, y)
                            
                    except:
                        continue

        if best_prob >= threshold:
            return {
                "filename": os.path.basename(banner_path),
                "probability": best_prob,
                "location": best_loc
            }
        return None
        
    except Exception as e:
        print(f"\nError processing banner '{banner_path}': {e}")
        return None

def print_help():
    print("""
Mystery Pic Banner Scanner

This script helps you find where the current Mystery Pic is hiding among banners.

Optional Flags:
  -e, --early     Stops scanning as soon as a match above the threshold is found.
  -f, --fast      Use ultra-fast scanning mode (may be slightly less accurate).
  -t, --threads   Number of worker processes (default: CPU count * 2).
  -h, --help      Displays this help message.

Requirements:
  Place exactly one image file (PNG or GIF) in the 'MysteryPic/' folder.
  If a GIF is present, it will be automatically converted to a 10x10 PNG for matching.
  Place PNG images of banners inside the 'MysteryPic/banners/' folder.

Examples:
  python MysteryPic.py -e -f
  python MysteryPic.py --early --fast --threads 32
  python MysteryPic.py --help

""")

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-e", "--early", action="store_true")
    parser.add_argument("-f", "--fast", action="store_true", help="Use ultra-fast scanning mode")
    parser.add_argument("-t", "--threads", type=int, default=None, help="Number of worker processes")
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
    template_gray = cv2.cvtColor(scaled_template, cv2.COLOR_BGR2GRAY).astype("float64")
    
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

    # Determine number of workers
    cpu_count = os.cpu_count() or 4
    max_workers = args.threads if args.threads else min(cpu_count * 2, len(banner_paths))
    
    print(f"Scanning {len(banner_paths)} banners with {max_workers} workers...")
    print(f"Mode: {'Ultra-Fast' if args.fast else 'Optimized'}")
    
    start_time = time.time()
    all_matches = []
    early_stop = args.early

    # Choose scanning function based on mode
    if args.fast:
        scan_func = scan_banner_ultra_fast
        scan_args = [(template_gray, path, 0.95) for path in banner_paths]
    else:
        scan_func = scan_banner_optimized
        chunk_size = 100  # Process 100 positions at a time
        scan_args = [(template_gray, path, 0.95, chunk_size) for path in banner_paths]

    # Use multiprocessing for maximum speed
    with mp.Pool(processes=max_workers) as pool:
        try:
            if early_stop:
                # For early stopping, we need to handle results as they come
                results = pool.imap_unordered(scan_func, scan_args)
                for i, result in enumerate(results):
                    print(f"  Done: {os.path.basename(banner_paths[i % len(banner_paths)])}")
                    if result:
                        print(f"    Match Found: {result['filename']} @ {result['location']} | Probability: {result['probability']:.4f}")
                        all_matches.append(result)
                        pool.terminate()  # Stop all processes
                        break
            else:
                # Process all banners
                results = []
                with tqdm(total=len(scan_args), desc="Scanning") as pbar:
                    for result in pool.imap_unordered(scan_func, scan_args):
                        results.append(result)
                        pbar.update(1)
                        if result:
                            tqdm.write(f"    Match Found: {result['filename']} @ {result['location']} | Probability: {result['probability']:.4f}")
                
                all_matches = [r for r in results if r is not None]
                
        except KeyboardInterrupt:
            print("\nScan interrupted by user.")
            pool.terminate()
            pool.join()
            return

    if all_matches:
        print("\n--- High Probability Matches ---")
        all_matches.sort(key=lambda m: m['probability'], reverse=True)
        for m in all_matches:
            print(f"{m['filename']}: {m['probability']:.4f} at {m['location']}")
    else:
        print("No matches found above threshold.")

    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed:.2f} seconds.")
    print(f"Speed: {len(banner_paths)/elapsed:.1f} banners/second")

if __name__ == "__main__":
    main()
