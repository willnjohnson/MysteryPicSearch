# Mystery Pic Search

This Python script helps you search for the hidden image in Neopets' Mystery Pic competition by scanning banners for visual similarity.

**Note:** You'll need to download the banner images yourself (e.g. https://www.drsloth.com/search/?category=18 AND https://www.neopets.com/~twidlybee). I recommend naming your images appropriately, like Aisha_Shopkeeper.png, Almost_Abandoned_Attic.png, etc.

***Use Responsibly:** I recommend only using this script, if you're aiming for the gold trophy **once**, or when it's later in the day (i.e. 30 minutes to 1 hour after the Mystery Pic competition starts) and you just want the regular reward.*

## Where to Store Images

* Place **exactly one** image (either `.png` or `.gif`) in the `MysteryPic/` folder.
* Place **.png banner images** in the `MysteryPic/banners/` folder. Look at previous competitions to figure out what category of banners to download (e.g. Shopkeeper banners).
  * **Recommendation:** Rename the banner images according to what the image is (you'll need to do your own research), so you know exactly what to guess.

## Features

* **GIF to PNG Conversion:** Automatically converts `.gif` Mystery Pics to `.png` and resizes to 10x10 pixels.
* **Similarity Scanning:** Uses SSIM to compare the Mystery Pic image against all banners in the folder.
* **Early Exit Option:** Optional `--early` flag stops scanning when a confident match is found.
* **Multi-core Processing:** Efficiently scans using all available CPU cores.
* **Smart 2-pass approach for time improvement (--fast flag):**
  * Pass 1: Coarse search with step=3 (fast coverage)
  * Pass 2: Fine search around promising areas (step=1)
    
## Requirements

Install Python dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
* Run the script from the command line:
 > python MysteryPic.py --fast

* To stop scanning once a strong match is found:
 > python MysteryPic.py --fast --early

## Output
Results will show in the terminal, including any matches above the threshold (default is 95% similarity). If found, you'll see the banner filename, match probability, and XY coordinates.

## License

This project is open-source and available under the MIT License.

**Disclaimer:** "Neopets" is a registered trademark of Neopets, Inc. This script is an unofficial fan-made helper and is not affiliated with or endorsed by Neopets, Inc.
