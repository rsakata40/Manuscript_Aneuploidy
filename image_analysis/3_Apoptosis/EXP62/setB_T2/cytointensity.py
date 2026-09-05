#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import pandas as pd
import tifffile as tiff

# ---------------- CONFIG ----------------
DIR_IMG = "/ceph.groups/mshahbazi.grp/rsakata/EXP62/LIF2TIF/setB_T2"         # images: Z,C,Y,X
DIR_NUC = "/ceph.groups/mshahbazi.grp/rsakata/EXP62/output/setB_T2/1_cellpose_mask"    # nuclear masks: Z,Y,X  (filename: <file>_cp_mask.tif/.tiff)
DIR_SPH = "/ceph.groups/mshahbazi.grp/rsakata/EXP62/output/setB_T2/2_cytoseg"            # spheroid masks: Z,Y,X (filename: <file>_mask.tif/.tiff)
OUT_DIR = "/ceph.groups/mshahbazi.grp/rsakata/EXP62/output/setB_T2/3_cytomask_intensities"
out_csv = os.path.join(OUT_DIR, f"setA_cyto_intensity.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# number of channels & names (in the order they appear along C)
N_CHANNELS    = 5
CHANNEL_NAMES = ["dapi", "phalloidin", "GFP", "ECAD", "caspase3"]   # edit to match your data

# If the file has extra channels, list which to measure (0-based). Otherwise None → [0..N_CHANNELS-1]
CHANNEL_INDICES = None
# ---------------------------------------

assert len(CHANNEL_NAMES) == N_CHANNELS, "CHANNEL_NAMES must match N_CHANNELS"

def find_path(base_dir, stem, suffix, exts=(".tif", ".tiff")):
    for ext in exts:
        p = os.path.join(base_dir, f"{stem}{suffix}{ext}")
        if os.path.exists(p):
            return p
    return None

def load_img_ZCYX(path):
    arr = tiff.imread(path)
    if arr.ndim != 4:
        raise ValueError(f"{os.path.basename(path)}: expected 4D (Z,C,Y,X), got {arr.shape}")
    return arr

def load_mask_ZYX(path):
    arr = tiff.imread(path)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"{os.path.basename(path)}: expected 3D mask (Z,Y,X), got {arr.shape}")
    return (arr > 0)

def channel_means_ZCYX(img_ZCYX, mask_ZYX, chan_idx):
    Z, C, Y, X = img_ZCYX.shape
    if mask_ZYX.shape != (Z, Y, X):
        raise ValueError(f"Mask shape {mask_ZYX.shape} ≠ image spatial shape {(Z,Y,X)}")
    m = mask_ZYX.reshape(-1)
    nvox = int(m.sum())
    if nvox == 0:
        return [np.nan] * len(chan_idx)
    means = []
    for c in chan_idx:
        vals = img_ZCYX[:, c, :, :].reshape(-1)[m]
        means.append(float(vals.mean()))
    return means

def main():
    records = []
    img_paths = sorted(glob.glob(os.path.join(DIR_IMG, "*.tif")) +
                       glob.glob(os.path.join(DIR_IMG, "*.tiff")))
    if not img_paths:
        print(f"No TIFFs found in {DIR_IMG}")
        return

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing {stem}")

        nuc_path = find_path(DIR_NUC, stem, "_cp_masks")
        sph_path = find_path(DIR_SPH, stem, "_mask")
        if nuc_path is None or sph_path is None:
            print(f"  WARNING: missing masks for {stem} (nuc={nuc_path}, sph={sph_path}); skipping.")
            continue

        try:
            img = load_img_ZCYX(img_path)     # (Z,C,Y,X)
            Z, C, Y, X = img.shape
            nuc = load_mask_ZYX(nuc_path)     # (Z,Y,X)
            sph = load_mask_ZYX(sph_path)     # (Z,Y,X)
        except Exception as e:
            print(f"  ERROR loading {stem}: {e}")
            continue

        # choose channels
        if CHANNEL_INDICES is None:
            chan_idx = list(range(N_CHANNELS))
        else:
            chan_idx = list(CHANNEL_INDICES)
            assert len(chan_idx) == N_CHANNELS, "CHANNEL_INDICES must match N_CHANNELS"
        if any(ci < 0 or ci >= C for ci in chan_idx):
            print(f"  ERROR: channel indices {chan_idx} out of bounds for C={C}; skipping.")
            continue

        # strict shape check
        if nuc.shape != (Z, Y, X) or sph.shape != (Z, Y, X):
            print(f"  ERROR: mask shapes nuc={nuc.shape}, sph={sph.shape} must both be {(Z,Y,X)}; skipping.")
            continue

        # cytomask = spheroid minus nuclear
        cyto = sph & (~nuc)

        # per-channel means
        means = channel_means_ZCYX(img, cyto, chan_idx)

        rec = {"file": stem, "Z": Z, "Y": Y, "X": X, "n_vox_cytomask": int(cyto.sum())}
        for name, val in zip(CHANNEL_NAMES, means):
            rec[f"{name}_mean"] = val
        records.append(rec)

    if not records:
        print("No measurements written.")
        return

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {OUT_DIR}")
    print(df.head(min(5, len(df))))

if __name__ == "__main__":
    main()
