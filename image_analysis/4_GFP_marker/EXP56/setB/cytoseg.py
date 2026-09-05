#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, glob
import natsort
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import filters, util
from skimage.morphology import reconstruction, disk, remove_small_holes
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
ROOT_DIR = '/ceph.groups/mshahbazi.grp/rsakata/EXP56/LIF2TIF/setB'
OUT_DIR  = '/ceph.groups/mshahbazi.grp/rsakata/EXP56/cellpose/cyto_mask/setB'

# Voxel size (µm/pixel)
z_spacing_um_per_px  = 2.0
xy_spacing_um_per_px = 0.7583174   # assumed isotropic in XY

# --- Channel handling (Z,C,Y,X) ---
DAPI_CH        = 0                 # shown as "Raw DAPI" in QC
MERGE_CHANNELS = [0, 1,2,3,4]  # channels to merge into pre_combined
MERGE_MODE     = "max"             # "max" or "sum"
MERGE_WEIGHTS  = None              # e.g. [1.0, 0.5, 2.0, ...] or None

# --- Preprocessing (per-channel before merging) ---
sigma_xy = 0.5
sigma_z  = 0.0
clip_low, clip_high = 1.0, 99.8
bg_sigma_xy = 50
bg_sigma_z  = 0
flatfield   = True
gamma       = 0.7
median_xy_px   = 5
open_xy_px_pre = 1      # grayscale opening per-XY slice (disk radius, Z-thickness=1)

# --- Thresholding ---
# One of: "fixed", "percentile", "otsu", "local", "sauvola", "hysteresis", "recon"
method        = "fixed"
fixed_t       = 0.0
percentile_p  = 99.5
percentile_per_slice = False
local_block   = 31
local_offset  = 0.0
sauvola_k     = 0.2
hys_low_rel   = 0.5
recon_low_rel = 0.5

# --- Per-slice post-threshold filters (applied to EVERY z-slice) ---
AREA_CONNECTIVITY_2D = 8   

# Remove small objects (µm²): set to None or <=0 to disable
MIN_OBJECT_AREA_UM2 = 400

# Remove large objects (µm²): set to None or <=0 to disable
MAX_OBJECT_AREA_UM2 = 50000

# Fill small holes inside objects (µm²): set to None or <=0 to disable
FILL_HOLES_MAX_UM2  = 3000

# --- QC image ---
DO_QC  = True
dpi_qc = 150
# ===================================================


# ===================== helpers =====================
def _rescale01(a):
    a = a.astype(np.float32, copy=False)
    lo, hi = np.percentile(a, (1, 99.8))
    if hi > lo: a = (a - lo) / (hi - lo + 1e-8)
    return np.clip(a, 0, 1, out=a)

def preprocess(vol_z_y_x,
               sigma_xy=1.0, sigma_z=0.0,
               clip_low=1.0, clip_high=99.8,
               *, flatfield=True, bg_sigma_xy=50, bg_sigma_z=0,
               per_slice_norm=True, gamma=None,
               median_xy_px=3, median_z_slices=0, open_xy_px=0):
    """Per-channel preprocessing; vol is (Z,Y,X)."""
    v = util.img_as_float32(vol_z_y_x)

    if flatfield:
        bg = ndi.gaussian_filter(v, sigma=(bg_sigma_z, bg_sigma_xy, bg_sigma_xy))
        v = v / (bg + 1e-8)

    if per_slice_norm:
        for z in range(v.shape[0]):
            lo, hi = np.percentile(v[z], (clip_low, clip_high))
            if hi > lo:
                v[z] = np.clip((v[z] - lo) / (hi - lo + 1e-8), 0, 1, out=v[z])
    else:
        lo, hi = np.percentile(v, (clip_low, clip_high))
        if hi > lo:
            v = np.clip((v - lo) / (hi - lo + 1e-8), 0, 1, out=v)

    if gamma and gamma > 0:
        v = np.power(v, gamma, dtype=np.float32)

    if (median_xy_px and median_xy_px > 0) or (median_z_slices and median_z_slices > 0):
        sz_z  = max(1, int(median_z_slices)) if median_z_slices else 1
        sz_xy = max(1, int(median_xy_px))    if median_xy_px    else 1
        v = ndi.median_filter(v, size=(sz_z, sz_xy, sz_xy))

    if open_xy_px and open_xy_px > 0:
        r = int(open_xy_px)
        se_xy = disk(r).astype(bool)
        se_3d = np.zeros((1, *se_xy.shape), dtype=bool)  # Z=1 → per-slice
        se_3d[0] = se_xy
        v = ndi.grey_opening(v, footprint=se_3d)

    if sigma_xy > 0 or sigma_z > 0:
        v = ndi.gaussian_filter(v, sigma=(sigma_z, sigma_xy, sigma_xy))

    return v

def combine_preprocessed(ch_vols, mode="max", weights=None):
    """ch_vols: list of (Z,Y,X) float32 volumes, already preprocessed."""
    if len(ch_vols) == 1:
        return ch_vols[0].astype(np.float32, copy=False)
    if mode == "sum":
        if weights is None:
            s = np.sum(ch_vols, axis=0, dtype=np.float32)
        else:
            assert len(weights) == len(ch_vols), "weights must match MERGE_CHANNELS"
            s = np.zeros_like(ch_vols[0], dtype=np.float32)
            for v, w in zip(ch_vols, weights):
                s += float(w) * v
        lo, hi = np.percentile(s, (1, 99.8))
        if hi > lo:
            s = np.clip((s - lo) / (hi - lo + 1e-8), 0, 1, out=s)
        return s
    # default max
    out = ch_vols[0].copy()
    for v in ch_vols[1:]:
        np.maximum(out, v, out=out)
    return out.astype(np.float32, copy=False)

def threshold_3d(vol,
                 method="otsu",
                 block_size=31, offset=0.0,
                 sauvola_k=0.2,
                 hys_low_rel=0.5,
                 recon_low_rel=0.5,
                 fixed_t=0.0,
                 percentile_p=99.5,
                 percentile_per_slice=False):
    """Return boolean mask (Z,Y,X)."""
    if method in {"fixed", "abs", "absolute"}:
        return (vol > fixed_t)

    if method in {"percentile", "perc", "quantile"}:
        if vol.ndim == 3 and percentile_per_slice:
            m = np.zeros_like(vol, dtype=bool)
            for z in range(vol.shape[0]):
                t = np.percentile(vol[z], percentile_p)
                m[z] = vol[z] >= t
            return m
        t = np.percentile(vol, percentile_p)
        return vol >= t

    if method in {"otsu", "global"}:
        th = filters.threshold_otsu(vol)
        return vol > th

    from skimage.filters import threshold_sauvola, apply_hysteresis_threshold
    if method == "local":
        m = np.zeros_like(vol, dtype=bool)
        for z in range(vol.shape[0]):
            tl = filters.threshold_local(vol[z], block_size=block_size, offset=offset)
            m[z] = vol[z] > tl
        return m
    if method == "sauvola":
        m = np.zeros_like(vol, dtype=bool)
        for z in range(vol.shape[0]):
            tl = threshold_sauvola(vol[z], window_size=block_size, k=sauvola_k)
            m[z] = vol[z] > tl
        return m
    if method == "hysteresis":
        high = filters.threshold_otsu(vol)
        low  = high * float(hys_low_rel)
        m = np.zeros_like(vol, dtype=bool)
        for z in range(vol.shape[0]):
            m[z] = apply_hysteresis_threshold(vol[z], low, high)
        return m
    if method == "recon":
        high = filters.threshold_otsu(vol)
        low  = high * float(recon_low_rel)
        seed  = (vol > high).astype(np.uint8)
        allow = (vol > low ).astype(np.uint8)
        rec = reconstruction(seed, allow, method='dilation')
        return rec.astype(bool)

    raise ValueError(f"Unknown method: {method}")

def _struct2d(connectivity=8):
    """2D structuring element for SciPy labeling."""
    if connectivity == 4:
        return ndi.generate_binary_structure(2, 1)  # 4-connected
    else:
        return np.ones((3, 3), dtype=bool)          # 8-connected

def _um2_to_px(area_um2, xy_um):
    if not area_um2 or area_um2 <= 0:
        return 0
    return max(1, int(round(float(area_um2) / (float(xy_um) ** 2))))

def filter_mask_per_slice(mask_bool_ZYX,
                          xy_um_per_px,
                          min_area_um2=0.0,
                          max_area_um2=0.0,
                          fill_holes_max_um2=0.0,
                          connectivity=8):
    """
    Apply per-slice object filtering + hole filling.
    - remove 2D objects < min_area_um2
    - remove 2D objects > max_area_um2
    - fill interior holes up to fill_holes_max_um2
    """
    Z, Y, X = mask_bool_ZYX.shape
    out = mask_bool_ZYX.copy()

    min_px = _um2_to_px(min_area_um2, xy_um_per_px)
    max_px = _um2_to_px(max_area_um2, xy_um_per_px)
    hole_px = _um2_to_px(fill_holes_max_um2, xy_um_per_px)
    struct2d = _struct2d(connectivity)

    for z in range(Z):
        m = out[z]
        if not m.any():
            continue

        # Fill small holes first (stabilizes object areas)
        if hole_px > 0:
            m = remove_small_holes(m, area_threshold=hole_px, connectivity=1 if connectivity==4 else 2)

        # Label 2D components
        lab2d, nlab = ndi.label(m, structure=struct2d)
        if nlab == 0:
            out[z] = m
            continue

        counts = np.bincount(lab2d.ravel())

        # Remove too-small
        if min_px > 0:
            too_small = np.where(counts < min_px)[0]
            too_small = too_small[too_small != 0]
            if too_small.size:
                m[ np.isin(lab2d, too_small) ] = False

        # Remove too-large
        if max_px > 0:
            counts = np.bincount(lab2d.ravel())  # recompute after small removal
            too_big = np.where(counts > max_px)[0]
            too_big = too_big[too_big != 0]
            if too_big.size:
                m[ np.isin(lab2d, too_big) ] = False

        out[z] = m

    return out

def save_bool_mask_tiff(path, mask_bool):
    tiff.imwrite(path, mask_bool.astype(np.uint8), imagej=True)

def show_qc_slices_top_mid_last(raw_dapi, pre_combined, mask_bool, out_png_path, dpi=150):
    Z = raw_dapi.shape[0]
    z_list = [0, Z//2, Z-1]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    cols = ["Raw DAPI", "Preprocessed (combined)", "Threshold mask"]
    for r, z in enumerate(z_list):
        raw_disp  = _rescale01(raw_dapi[z])
        pre_disp  = _rescale01(pre_combined[z])
        mask_disp = mask_bool[z].astype(float)
        panels = [raw_disp, pre_disp, mask_disp]
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(panels[c], cmap='gray')
            if r == 0: ax.set_title(cols[c])
            ax.set_xlabel(f"z={z}")
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_png_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# ===================== main =====================
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(ROOT_DIR, "*.tif")) +
        glob.glob(os.path.join(ROOT_DIR, "*.tiff")),
        key=lambda x: natsort.natsort_key(x)
    )
    print(f"Found {len(files)} TIFs.")

    # Optional SLURM array
    tid = int(os.getenv("SLURM_ARRAY_TASK_ID", "-1"))
    if tid >= 0:
        if tid >= len(files):
            print(f"[array] task {tid} out of range (0..{len(files)-1}). Exiting.")
            return
        files = [files[tid]]
        print(f"[array] task {tid}: {files[0]}")

    for fpath in files:
        fname = os.path.basename(fpath)
        fbase = fname.rsplit('.', 1)[0]
        print(f"\n=== Processing: {fname} ===")

        vol = tiff.imread(fpath)  # expect (Z, C, Y, X)
        if vol.ndim != 4:
            print(f"  ERROR: expected (Z,C,Y,X), got {vol.shape}; skipping.")
            continue
        Z, C, Y, X = vol.shape

        # channel sanity
        if any(ch < 0 or ch >= C for ch in MERGE_CHANNELS):
            print(f"  ERROR: MERGE_CHANNELS {MERGE_CHANNELS} out of bounds for C={C}; skipping.")
            continue
        if not (0 <= DAPI_CH < C):
            print(f"  ERROR: DAPI_CH {DAPI_CH} out of bounds for C={C}; skipping.")
            continue

        # QC raw channel
        raw_dapi = util.img_as_float32(vol[:, DAPI_CH, :, :])

        # Preprocess selected channels then merge
        ch_vols = []
        for ch in MERGE_CHANNELS:
            v = preprocess(vol[:, ch, :, :],
                           sigma_xy=sigma_xy, sigma_z=sigma_z,
                           clip_low=clip_low, clip_high=clip_high,
                           flatfield=flatfield, bg_sigma_xy=bg_sigma_xy, bg_sigma_z=bg_sigma_z,
                           per_slice_norm=True, gamma=gamma,
                           median_xy_px=median_xy_px, median_z_slices=0,
                           open_xy_px=open_xy_px_pre)
            ch_vols.append(v)
        pre_combined = combine_preprocessed(ch_vols, mode=MERGE_MODE, weights=MERGE_WEIGHTS)

        # Threshold to binary
        mask = threshold_3d(
            pre_combined, method=method,
            block_size=local_block, offset=local_offset,
            sauvola_k=sauvola_k, hys_low_rel=hys_low_rel, recon_low_rel=recon_low_rel,
            fixed_t=fixed_t, percentile_p=percentile_p,
            percentile_per_slice=percentile_per_slice
        ).astype(bool)

        # Per-slice object filtering and hole filling (no head-slice restriction)
        mask = filter_mask_per_slice(
            mask_bool_ZYX=mask,
            xy_um_per_px=xy_spacing_um_per_px,
            min_area_um2=MIN_OBJECT_AREA_UM2,
            max_area_um2=MAX_OBJECT_AREA_UM2,
            fill_holes_max_um2=FILL_HOLES_MAX_UM2,
            connectivity=AREA_CONNECTIVITY_2D
        )

        # Save binary mask
        out_mask_path = os.path.join(OUT_DIR, f"{fbase}_mask.tif")
        save_bool_mask_tiff(out_mask_path, mask)
        print(f"  Saved mask: {out_mask_path}  (voxels={int(mask.sum())})")

        # QC (top/middle/last × raw/pre/mask)
        if DO_QC:
            out_png = os.path.join(OUT_DIR, f"{fbase}_qc.png")
            show_qc_slices_top_mid_last(raw_dapi, pre_combined, mask, out_png_path=out_png, dpi=dpi_qc)
            print(f"  Saved QC: {out_png}")

    print(f"\nDone in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
