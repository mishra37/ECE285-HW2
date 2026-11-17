"""Resize WikiArt images once to 64x64 and cache them as a uint8 .npy array.

WikiArt originals are large JPEGs; decoding them every epoch is very slow, so we
do it once. After this, the raw dataset can be deleted to free disk space.

Usage:
  python preprocess.py --root data/wikiart --out data/wikiart64.npy
  python preprocess.py --root data/wikiart --styles Impressionism --out data/impr64.npy
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load(args):
    path, size = args
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = min(w, h)  # center crop to square, then resize
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        img = img.resize((size, size), Image.BICUBIC)
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--styles", nargs="*", default=None, help="subfolder names to keep (default: all)")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    paths = []
    for d, _, files in os.walk(a.root):
        if a.styles and os.path.basename(d) not in a.styles:
            continue
        paths += [os.path.join(d, f) for f in files if os.path.splitext(f)[1].lower() in EXTS]
    paths.sort()
    if a.max_images:
        rng = np.random.default_rng(0)
        paths = list(rng.choice(paths, size=min(a.max_images, len(paths)), replace=False))
    print(f"Found {len(paths)} images")

    with ProcessPoolExecutor(a.workers) as ex:
        imgs = list(tqdm(ex.map(load, [(p_, a.size) for p_ in paths], chunksize=64), total=len(paths)))
    imgs = np.stack([im for im in imgs if im is not None])
    np.save(a.out, imgs)
    print(f"Saved {imgs.shape} -> {a.out}")


if __name__ == "__main__":
    main()
