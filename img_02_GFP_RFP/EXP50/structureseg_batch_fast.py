#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Creating blastoid masks — "strict-compat" speedup:
# - identical algorithms/parameters to your original
# - faster by avoiding Python loops & extra copies
# - QC plotting off by default

import os, math, json, time
import glob, natsort
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, util
from skimage.feature import peak_local_max
from skimage.morphology import reconstruction, ball, disk
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt

# ============== Config (same defaults as your original) ==============
ROOT_DIR = '/ceph.groups/mshahbazi.grp/rsakata/EXP50/LIF2TIF'   
outdir   = '/ceph.groups/mshahbazi.grp/rsakata/EXP50/cellpose_withstruc/mask'
remove_from_file_name = ""

# microns per pixel (your values)
z_spacing_um_per_px  = 2
xy_spacing_um_per_px = 0.3787879

# Preprocessing
sigma_xy = 0.5
sigma_z  = 0.0
clip_low, clip_high = 1.0, 99.8
bg_sigma_xy = 50
bg_sigma_z  = 0
flatfield   = True
gamma       = 0.7
median_xy_px   = 5
open_xy_px_pre = 1  # grayscale opening (disk) per XY slice

# Thresholding (keep your choice)
method  = "fixed"
fixed_t = 0

# Binary cleanup
open_xy_px_mask = 2
close_xy        = 1
min_size_vox    = 50000

# Split & size filter
min_peak_dist_um = 40
min_diam_um = 50.0
max_diam_um = 1000.0

# QC
DO_QC = True
z_idx_for_qc = None

# ============== Utils (unchanged semantics) ==============
def sample_from_filename(fname_base, remove_prefix=None):
    s = fname_base
    if s.lower().endswith('.tiff'): s = s[:-5]
    elif s.lower().endswith('.tif'): s = s[:-4]
    if ' - ' in s: return s.split(' - ', 1)[1].strip()
    if remove_prefix and s.startswith(remove_prefix): s = s[len(remove_prefix):]
    return s.strip()

def load_channels_ZCYX(path, ch_dapi=0, ch_phall=1):
    vol = tiff.imread(path)
    if vol.ndim != 4 or not (vol.shape[1] <= 8 and vol.shape[2] > 8 and vol.shape[3] > 8):
        raise ValueError(f"Expected ZCYX (Z,C,Y,X). Got {vol.shape}")
    # identical conversion to float32
    dapi = util.img_as_float32(vol[:, ch_dapi, :, :])
    phal = util.img_as_float32(vol[:, ch_phall, :, :])
    return dapi, phal

# ============== Strict-compatible preprocessing (no algorithm change) ==============
def preprocess(
    vol,
    sigma_xy=1.0,
    sigma_z=0.0,
    clip_low=1.0,
    clip_high=99.8,
    *,
    flatfield=True,
    bg_sigma_xy=50,
    bg_sigma_z=0,
    per_slice_norm=False,
    gamma=None,
    median_xy_px=3,
    median_z_slices=0,
    open_xy_px=0
):
    # same dtype/scale path
    v = util.img_as_float32(vol)

    # (1) Flat-field: original Gaussian at full res (no approximation)
    if flatfield:
        bg = ndi.gaussian_filter(v, sigma=(bg_sigma_z, bg_sigma_xy, bg_sigma_xy))
        v = v / (bg + 1e-8)

    # (2) Robust normalization by true percentiles (same as original)
    if per_slice_norm and v.ndim == 3:
        for z in range(v.shape[0]):
            lo, hi = np.percentile(v[z], (clip_low, clip_high))
            if hi <= lo:  # guard
                continue
            v[z] = np.clip((v[z] - lo) / (hi - lo + 1e-8), 0, 1, out=v[z])
    else:
        lo, hi = np.percentile(v, (clip_low, clip_high))
        if hi > lo:
            v = np.clip((v - lo) / (hi - lo + 1e-8), 0, 1, out=v)

    # (3) Gamma (same)
    if gamma is not None and gamma > 0:
        v = np.power(v, gamma, dtype=np.float32)

    # (4) Median filter (same)
    if (median_xy_px and median_xy_px > 0) or (median_z_slices and median_z_slices > 0):
        sz_z  = max(1, int(median_z_slices)) if median_z_slices else 1
        sz_xy = max(1, int(median_xy_px))    if median_xy_px    else 1
        v = ndi.median_filter(v, size=(sz_z, sz_xy, sz_xy))

    # (5) Grayscale opening per XY slice — **result-identical** but faster:
    #     Use a 3D footprint that is 1 voxel in Z and a disk in XY.
    if open_xy_px and open_xy_px > 0:
        r = int(open_xy_px)
        se_xy = disk(r).astype(bool)                 # 2D disk (same as original)
        se_3d = np.zeros((1, se_xy.shape[0], se_xy.shape[1]), dtype=bool)
        se_3d[0] = se_xy                             # thickness 1 in Z → per-slice opening
        v = ndi.grey_opening(v, footprint=se_3d)     # single vectorized call

    # (6) Mild Gaussian denoise (same)
    if sigma_xy > 0 or sigma_z > 0:
        v = ndi.gaussian_filter(v, sigma=(sigma_z, sigma_xy, sigma_xy))

    return v

def combine_channels(pre_dapi, pre_phal, mode="max", w_dapi=1.0, w_phal=1.0):
    if mode == "dapi":  return pre_dapi
    if mode == "phall": return pre_phal
    if mode == "sum":
        s = w_dapi*pre_dapi + w_phal*pre_phal
        lo, hi = np.percentile(s, (1, 99.8))
        if hi > lo:
            s = np.clip((s - lo) / (hi - lo + 1e-8), 0, 1, out=s)
        return s.astype(np.float32, copy=False)
    # default 'max' (same)
    return np.maximum(pre_dapi, pre_phal).astype(np.float32, copy=False)

# ============== Thresholding (unchanged) ==============
def threshold_3d(
    vol,
    method="otsu",
    block_size=31,
    offset=0.0,
    sauvola_k=0.2,
    hys_low_rel=0.5,
    recon_low_rel=0.5,
    fixed_t=0,
    percentile_p=99.5,
    percentile_per_slice=False
):
    if method in {"fixed", "abs", "absolute"}:
        return vol > fixed_t

    if method in {"percentile", "perc", "quantile"}:
        if vol.ndim == 3 and percentile_per_slice:
            mask = np.zeros_like(vol, dtype=bool)
            for z in range(vol.shape[0]):
                t = np.percentile(vol[z], percentile_p)
                mask[z] = vol[z] >= t
            return mask
        t = np.percentile(vol, percentile_p)
        return vol >= t

    if method in {"otsu", "global"}:
        th = filters.threshold_otsu(vol)
        return vol > th

    from skimage.filters import threshold_sauvola, apply_hysteresis_threshold

    if method == "local":
        if vol.ndim == 2:
            tl = filters.threshold_local(vol, block_size=block_size, offset=offset)
            return vol > tl
        mask = np.zeros_like(vol, dtype=bool)
        for z in range(vol.shape[0]):
            tl = filters.threshold_local(vol[z], block_size=block_size, offset=offset)
            mask[z] = vol[z] > tl
        return mask

    if method == "sauvola":
        if vol.ndim == 2:
            tl = threshold_sauvola(vol, window_size=block_size, k=sauvola_k)
            return vol > tl
        mask = np.zeros_like(vol, dtype=bool)
        for z in range(vol.shape[0]):
            tl = threshold_sauvola(vol[z], window_size=block_size, k=sauvola_k)
            mask[z] = vol[z] > tl
        return mask

    if method == "hysteresis":
        high = filters.threshold_otsu(vol)
        low  = high * float(hys_low_rel)
        if vol.ndim == 2:
            return apply_hysteresis_threshold(vol, low, high)
        mask = np.zeros_like(vol, dtype=bool)
        for z in range(vol.shape[0]):
            mask[z] = apply_hysteresis_threshold(vol[z], low, high)
        return mask

    if method == "recon":
        high = filters.threshold_otsu(vol)
        low  = high * float(recon_low_rel)
        seed  = (vol > high).astype(np.uint8)
        allow = (vol > low ).astype(np.uint8)
        rec = reconstruction(seed, allow, method='dilation')
        return rec.astype(bool)

    raise ValueError("Unknown method")

# ============== Binary clean (result-identical per-slice) ==============
def fill_and_clean(mask, close_xy=3, min_size_vox=50, open_xy_px=0):
    m = mask.astype(bool, copy=False)

    # Opening per slice with a Z=1, XY=disk footprint (identical to original loop)
    if open_xy_px and open_xy_px > 0:
        r = int(open_xy_px)
        se_xy = disk(r).astype(bool)
        se_3d = np.zeros((1, se_xy.shape[0], se_xy.shape[1]), dtype=bool)
        se_3d[0] = se_xy
        m = ndi.binary_opening(m, structure=se_3d)

    # Fill holes per slice but vectorized: run once with Z=1 footprint
    se_fill = np.ones((1, 3, 3), dtype=bool)  # connectivity within each slice
    m = ndi.binary_fill_holes(m, structure=se_fill)

    # Closing per slice (same idea)
    if close_xy and close_xy > 0:
        r = int(close_xy)
        se_xy = disk(r).astype(bool)
        se_3d = np.zeros((1, se_xy.shape[0], se_xy.shape[1]), dtype=bool)
        se_3d[0] = se_xy
        m = ndi.binary_closing(m, structure=se_3d)

    # Remove small objects in 3D with connectivity=3 (same as original)
    m = morphology.remove_small_objects(m, min_size=min_size_vox, connectivity=3)
    return m

# ============== Split & measure (unchanged math) ==============
def watershed_split(mask, z_um, xy_um, min_peak_dist_um=30):
    sampling = (z_um, xy_um, xy_um)
    dt = ndi.distance_transform_edt(mask, sampling=sampling)

    # Original full-res peak search via physical footprint
    dxy = max(1, int(round(min_peak_dist_um / xy_um)))
    dz  = max(1, int(round(min_peak_dist_um / z_um)))
    footprint = np.ones((dz, dxy, dxy), dtype=bool)
    peaks = peak_local_max(dt, footprint=footprint, labels=mask, exclude_border=False)

    markers = np.zeros_like(mask, dtype=np.int32)
    for i, (zz, yy, xx) in enumerate(peaks, start=1):
        markers[zz, yy, xx] = i
    if markers.max() == 0:
        markers, _ = ndi.label(mask)

    labels = segmentation.watershed(-dt, markers, mask=mask)
    return dt, markers, labels

def v_um3_from_diam_um(d_um): return (math.pi/6.0) * (d_um**3)

def volume_filter(labels, z_um, xy_um, min_diam_um=40, max_diam_um=1000):
    vox_um3 = z_um * (xy_um**2)
    min_vox = int(round(v_um3_from_diam_um(min_diam_um) / vox_um3))
    max_vox = int(round(v_um3_from_diam_um(max_diam_um) / vox_um3))
    out = np.zeros_like(labels, dtype=np.int32)
    labs = np.unique(labels); labs = labs[labs != 0]
    k = 1
    for lab in labs:
        sz = np.count_nonzero(labels == lab)
        if min_vox <= sz <= max_vox:
            out[labels == lab] = k; k += 1
    return out

def save_label_tiff(outdir, labels, fbase):
    out_path = os.path.join(outdir, f"{fbase}_mask.tif")
    tiff.imwrite(out_path, labels.astype(np.uint16))

def remove_xy_edge_touching(labels, margin_px=0, max_touch_px=0, relabel=True):
    if labels.ndim != 3: raise ValueError("labels must be 3D")
    labmax = int(labels.max())
    if labmax == 0: return labels
    pad = max(1, int(margin_px))
    band = np.zeros_like(labels, dtype=bool)
    band[:, :pad, :]  = True; band[:, -pad:, :] = True
    band[:, :, :pad]  = True; band[:, :, -pad:] = True
    touch_count = np.bincount(labels[band].ravel(), minlength=labmax+1)
    keep = np.ones(labmax+1, dtype=bool); keep[0] = False
    for lab in range(1, labmax+1):
        keep[lab] = (touch_count[lab] <= int(max_touch_px))
    out = labels.copy()
    bad = np.nonzero(~keep)[0]
    if bad.size: out[np.isin(out, bad)] = 0
    return segmentation.relabel_sequential(out)[0] if relabel else out

def remove_xy_edge_touching_um(labels, xy_um_per_px, margin_um=0.0, max_touch_um=0.0, relabel=True):
    margin_px    = int(math.ceil(float(margin_um)    / float(xy_um_per_px))) if margin_um    > 0 else 0
    max_touch_px = int(math.ceil(float(max_touch_um) / float(xy_um_per_px))) if max_touch_um > 0 else 0
    return remove_xy_edge_touching(labels, margin_px=margin_px, max_touch_px=max_touch_px, relabel=relabel)

# ============== QC helper (unchanged; disabled by default) ==============
def _rescale01(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99.8))
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)

def show_pipeline_slice(z, raw, pre, mask, cleaned, dt, markers, labels, labels_filt,
                        figsize=(18,10), alpha=0.35, random_seed=0,
                        outdir=outdir, dpi=300, show=True, fbase="slice"):
    from skimage.color import label2rgb
    if random_seed is not None: np.random.seed(random_seed)
    dt_slice   = dt[z] if dt is not None else np.zeros_like(raw[z])
    dt_disp    = _rescale01(dt_slice)
    raw_disp   = _rescale01(raw[z])
    pre_disp   = _rescale01(pre[z])
    mask_disp  = mask[z].astype(float)
    clean_disp = cleaned[z].astype(float)
    lab_slice  = labels[z]
    labf_slice = labels_filt[z]
    lab_rgb  = label2rgb(lab_slice,  image=pre_disp, bg_label=0, alpha=alpha, kind='overlay')
    labf_rgb = label2rgb(labf_slice, image=pre_disp, bg_label=0, alpha=alpha, kind='overlay')

    fig, axes = plt.subplots(2, 4, figsize=figsize); ax = axes.ravel()
    ax[0].imshow(raw_disp, cmap='gray'); ax[0].set_title(f"Raw (z={z})"); ax[0].axis('off')
    ax[1].imshow(pre_disp, cmap='gray'); ax[1].set_title("Preprocessed"); ax[1].axis('off')
    ax[2].imshow(mask_disp, cmap='gray'); ax[2].set_title("Threshold mask"); ax[2].axis('off')
    ax[3].imshow(clean_disp, cmap='gray'); ax[3].set_title("Filled/cleaned"); ax[3].axis('off')
    ax[4].imshow(dt_disp, cmap='gray');   ax[4].set_title("Distance transform"); ax[4].axis('off')
    ax[5].imshow(pre_disp, cmap='gray');  ax[5].set_title("Markers (peaks)"); ax[5].axis('off')
    if markers is not None and np.any(markers):
        m2d = markers[z] if markers.ndim == 3 else markers
        yy, xx = np.nonzero(m2d)
        if yy.size: ax[5].plot(xx, yy, '.', markersize=3)
    ax[6].imshow(lab_rgb);  ax[6].set_title("Watershed labels (colored)");     ax[6].axis('off')
    from skimage.segmentation import find_boundaries
    b = find_boundaries(lab_slice, mode='inner');  ax[6].contour(b, levels=[0.5])
    ax[7].imshow(labf_rgb); ax[7].set_title("Size-filtered labels (colored)"); ax[7].axis('off')
    bf = find_boundaries(labf_slice, mode='inner'); ax[7].contour(bf, levels=[0.5])
    plt.tight_layout()
    if outdir:
        out_path = os.path.join(outdir, f"{fbase}_mask_slice.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved QC figure to: {out_path}")
    if show: plt.show()
    else: plt.close(fig)
    return fig

# ============== Main ==============
def main():
    t0 = time.time()
    files = sorted(
        [p for p in glob.glob(os.path.join(ROOT_DIR, "*.tif")) + glob.glob(os.path.join(ROOT_DIR, "*.tiff"))],
        key=lambda x: natsort.natsort_key(x)
    )

    # Slurm array support (one file per task)
    tid = int(os.getenv("SLURM_ARRAY_TASK_ID", "-1"))
    if tid >= 0:
        if tid < 0 or tid >= len(files):
            print(f"[array] task {tid} out of range (0..{len(files)-1}). Exiting.")
            return
        files = [files[tid]]
        print(f"[array] task {tid}: {files[0]}")

    print(f"Found {len(files)} TIFs.")
    os.makedirs(outdir, exist_ok=True)

    for fpath in files:
        fname = os.path.basename(fpath) 
        fbase = fname[:-4]
        sample = sample_from_filename(fname, remove_from_file_name)
        print("\n=== Processing:", fname, "===")

        # Load
        raw_dapi, raw_phal = load_channels_ZCYX(fpath, ch_dapi=0, ch_phall=1)

        # Preprocess (strict-compatible)
        pre_dapi = preprocess(raw_dapi,
                              sigma_xy=sigma_xy, sigma_z=sigma_z,
                              clip_low=clip_low, clip_high=clip_high,
                              flatfield=flatfield, bg_sigma_xy=bg_sigma_xy, bg_sigma_z=bg_sigma_z,
                              per_slice_norm=True, gamma=gamma,
                              median_xy_px=median_xy_px, median_z_slices=0, open_xy_px=open_xy_px_pre)
        pre_phal = preprocess(raw_phal,
                              sigma_xy=sigma_xy, sigma_z=sigma_z,
                              clip_low=clip_low, clip_high=clip_high,
                              flatfield=flatfield, bg_sigma_xy=bg_sigma_xy, bg_sigma_z=bg_sigma_z,
                              per_slice_norm=True, gamma=gamma,
                              median_xy_px=median_xy_px, median_z_slices=0, open_xy_px=open_xy_px_pre)

        # Combine (same)
        pre_combined = combine_channels(pre_dapi, pre_phal, mode="max")

        # Threshold (same)
        if method == "recon":
            mask = threshold_3d(pre_combined, method="recon", recon_low_rel=0.5)
        elif method == "hysteresis":
            mask = threshold_3d(pre_combined, method="hysteresis", hys_low_rel=0.5)
        elif method == "local":
            mask = threshold_3d(pre_combined, method="local", block_size=31, offset=0.0)
        elif method == "otsu":
            mask = threshold_3d(pre_combined, method="otsu")
        elif method == "percentile":
            mask = threshold_3d(pre_combined, method="percentile", percentile_p=99.5)
        else:
            mask = threshold_3d(pre_combined, method="fixed", fixed_t=fixed_t)

        # Clean (per-slice semantics preserved)
        mask_clean = fill_and_clean(mask, open_xy_px=open_xy_px_mask, close_xy=close_xy, min_size_vox=min_size_vox)
        print("mask_clean voxels:", int(mask_clean.sum()))

        # EDT + watershed (full-res, identical logic)
        dt, markers, labels = watershed_split(mask_clean, z_um=z_spacing_um_per_px, xy_um=xy_spacing_um_per_px,
                                              min_peak_dist_um=min_peak_dist_um)
        print("dt max (um):", float(dt.max()))

        labels_filt = volume_filter(labels, z_um=z_spacing_um_per_px, xy_um=xy_spacing_um_per_px,
                                    min_diam_um=min_diam_um, max_diam_um=max_diam_um)

        labels_filt = remove_xy_edge_touching_um(labels_filt, xy_um_per_px=xy_spacing_um_per_px,
                                                 margin_um=0, max_touch_um=100.0)

        save_label_tiff(outdir, labels_filt, fbase=fbase)

        Z = pre_combined.shape[0]
        z_show = Z//2 if z_idx_for_qc is None else int(z_idx_for_qc)

        print("Labels before size filter:", int(labels.max()))
        print("Labels after size filter:",  int(labels_filt.max()))

        if DO_QC:
            show_pipeline_slice(z_show, raw_dapi, pre_combined, mask, mask_clean, dt, markers, labels, labels_filt,
                                figsize=(18,10), alpha=0.35, random_seed=0,
                                outdir=outdir, dpi=150, show=False, fbase=fbase)

    print(f"\nAll done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
