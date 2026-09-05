#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Creating blastoid masks

import os, math, json
import glob, natsort
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, exposure, util
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola, apply_hysteresis_threshold
from skimage.morphology import reconstruction

# === EDIT THESE ===
ROOT_DIR = '/ceph.groups/mshahbazi.grp/rsakata/EXP50/LIF2TIF'  # <— change
outdir = '/ceph.groups/mshahbazi.grp/rsakata/EXP50/cellpose_withstruc/mask'
remove_from_file_name = ""

channel   = 1       

# Provide either µm/px OR px/µm (we'll convert px/µm to µm/px automatically)
z_spacing_um_per_px  = 2    # e.g., 2.0 µm/px along Z  (put your value here)
xy_spacing_um_per_px = 0.3787879  # e.g., 0.758 µm/px along XY


# =========================
# Preprocessing (intensity)
# =========================
sigma_xy = 0.5         # Gaussian blur (pixels) in XY to denoise. 0.5–1.5 typical. ↑ = smoother, softer edges.
sigma_z  = 0.0         # Gaussian blur along Z (slices). Keep 0 unless Z is very noisy; try 0.5–1 if needed.

clip_low, clip_high = 1.0, 99.8
# Robust normalize by percentiles. Pulls intensities into [0..1] after trimming extremes.
# If background still uneven, combine with flat-field below.

bg_sigma_xy = 50
bg_sigma_z  = 0
flatfield = True  #Flat-field scale (Gaussian sigma, in pixels/slices) for illumination correction.
# Pick bg_sigma_xy ~ 0.5–0.8 × spheroid diameter (in px). Z usually 0.

gamma = 0.7
# Gamma correction on normalized image. <1 brightens dim signal (0.7–0.9 common), >1 compresses highlights.

median_xy_px = 5
# Median filter window (pixels). Great for salt/pepper speckles. 3–5 common.
# Note: this is the WINDOW SIZE (odd), not a radius.

open_xy_px_pre = 1
# Grayscale opening per slice (disk radius in pixels). 1–2 removes tiny bright flecks before thresholding.
# Set 0 to disable. If edges thin too much, reduce or turn off.

# =================
# Thresholding step
# =================
# Option A: global/local
# method = "otsu"                     # Fast global threshold. Good when illumination is even.
# block_size = 41                     # For "local": odd window (≈ 2–4× object diameter in px).
# offset = -0.05                      # For "local": shift threshold; negative keeps more dim pixels.

# Option B: hysteresis (dim rims connected to bright cores)
# method = "hysteresis"
# hys_low_rel = 0.6                   # Keep pixels above low=0.4–0.6×high (high = global Otsu). Lower -> more kept.

# Option C: geodesic reconstruction (dim but 3D-connected regions)
#method = "recon"
#recon_low_rel = 0.5                   # 0.4–0.6 typical. Lower keeps more faint, connected voxels.

# Option D: geodesic reconstruction (dim but 3D-connected regions)
#method = "percentile"
#percentile_p=99.5                  # percentile from 0-100 (for bright speckles)

# Option E: simple fixed threshold
method = "fixed"
fixed_t=0 # absolute threshold in image units


# =========================
# Fill & clean (binary mask)
# =========================
open_xy_px_mask = 2
# Binary opening per slice (disk radius in pixels). Removes tiny dots/filaments after threshold.
# 1–2 typical. If small features vanish, reduce.

close_xy = 1
# Binary closing per slice (disk radius). Bridges small gaps and smooths edges. 2–5 typical.
# If objects start merging, decrease.

min_size_vox = 50000  #around 30um (differs according to um/pixel)

# =================
# Splitting (3D DT)
# =================
min_peak_dist_um = 40  #40 for spheroids
# Minimum center spacing (µm) for watershed markers. Bigger = fewer splits (good if over-segmenting).
# Try 60–120 depending on expected spheroid spacing.

# =========================
# Size filter (post-split) 
# =========================
min_diam_um = 80.0
max_diam_um = 1000.0
# Keep labels whose equivalent-sphere diameter (µm) falls within this range.
# Set to your plausible spheroid size bounds.

# ===
# QC
# ===
z_idx_for_qc = None
# Which Z slice to preview in the QC panel. None = mid-slice. Set an int 0..Z-1 to force a specific slice.



# Utils ------------------------------------------------------------------
def sample_from_filename(fname_base, remove_prefix=None):
    """
    Return the sample name as the substring after ' - ' (space-hyphen-space),
    excluding the file extension. If no ' - ' is found, optionally strip a
    leading prefix (for backward compatibility), otherwise return the stem.
    """
    # strip extension (.tif/.tiff/.ome.tif handled like your previous code)
    s = fname_base
    if s.lower().endswith('.tiff'):
        s = s[:-5]
    elif s.lower().endswith('.tif'):
        s = s[:-4]

    # prefer everything after the first ' - '
    if ' - ' in s:
        return s.split(' - ', 1)[1].strip()

    # fallback: old behavior
    if remove_prefix and s.startswith(remove_prefix):
        s = s[len(remove_prefix):]

    return s.strip()
    
def to_um_per_pixel(val):
    """Return µm/px from either µm/px or px/µm input.
    If val > 1.5 we assume it's px/µm and return 1/val. Otherwise we assume µm/px.
    """
    if val is None:
        return None
    return 1.0/val if val > 1.5 else float(val)

def load_volume(path, channel=None):
    vol = tiff.imread(path)

    if vol.ndim == 3:
        # Already (Z, Y, X) single-channel
        return util.img_as_float32(vol)

    if vol.ndim == 4:
        # Expect ZCYX: (Z, C, Y, X)
        if vol.shape[1] <= 8 and vol.shape[2] > 8 and vol.shape[3] > 8:
            if channel is None:
                raise ValueError("Image is 4D ZCYX (Z,C,Y,X). Please set channel index.")
            v = vol[:, channel, :, :]        # -> (Z, Y, X)
            return util.img_as_float32(v)
        else:
            raise ValueError(f"Expected ZCYX (Z,C,Y,X). Got shape {vol.shape}.")

    raise ValueError(f"Unsupported image shape {vol.shape}; expected 3D (Z,Y,X) or 4D ZCYX (Z,C,Y,X).")


def combine_channels_ZCYX(path, channels, mode="max", weights=None, rescale=True):
    """
    Load ZCYX (Z,C,Y,X) TIFF and combine selected channels into one (Z,Y,X) volume.

    channels : int or list/tuple of ints (0-based channel indices)
    mode     : 'max' (default), 'sum', 'mean'
    weights  : optional weights for 'sum'/'mean' (len == n_channels)
    rescale  : rescale output to [0,1] using robust percentiles
    """
    vol = tiff.imread(path)                         # (Z,C,Y,X)
    if vol.ndim != 4:
        raise ValueError(f"Expected ZCYX (Z,C,Y,X). Got {vol.shape}")

    Z, C, Y, X = vol.shape
    idx = np.atleast_1d(channels).astype(int)
    if np.any((idx < 0) | (idx >= C)):
        raise ValueError(f"Channel indices {idx.tolist()} out of range 0..{C-1}")

    # Pull channels and convert to float32 in [0,1] (dtype-aware)
    sel = util.img_as_float32(vol[:, idx, :, :])    # (Z, Csel, Y, X)

    # Combine
    if sel.shape[1] == 1:
        out = sel[:, 0, :, :]                       # (Z,Y,X)
    elif mode == "max":
        out = np.max(sel, axis=1)
    elif mode in ("sum", "mean"):
        w = np.ones(sel.shape[1], np.float32) if weights is None else np.asarray(weights, np.float32)
        if w.size != sel.shape[1]:
            raise ValueError(f"weights length {w.size} != number of channels {sel.shape[1]}")
        out = np.tensordot(w, sel, axes=(0, 1))     # (Z,Y,X)
        if mode == "mean":
            out /= (w.sum() + 1e-8)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Optional robust rescale to [0,1]
    out = out.astype(np.float32)
    if rescale:
        lo, hi = np.percentile(out, (1, 99.8))
        out = np.clip((out - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
    return out


def preprocess(
    vol,
    sigma_xy=1.0,
    sigma_z=0.0,
    clip_low=1,
    clip_high=99.8,
    *,
    flatfield=True,
    bg_sigma_xy=50,
    bg_sigma_z=0,
    per_slice_norm=False,
    gamma=None,
    # --- NEW despeckle knobs ---
    median_xy_px=3,        # 0 = off; 3–5 removes tiny specks but keeps edges
    median_z_slices=0,     # 0 = off; 1 if you have salt/pepper along Z
    open_xy_px=0           # 0 = off; grayscale opening per slice; 1–2 shrinks tiny bright flecks
):
    v = util.img_as_float32(vol)

    if flatfield:
        bg = ndi.gaussian_filter(v, sigma=(bg_sigma_z, bg_sigma_xy, bg_sigma_xy))
        v = v / (bg + 1e-8)

    # Robust normalization
    if per_slice_norm:
        for z in range(v.shape[0]):
            lo, hi = np.percentile(v[z], (clip_low, clip_high))
            v[z] = np.clip((v[z] - lo) / (hi - lo + 1e-8), 0, 1)
    else:
        lo, hi = np.percentile(v, (clip_low, clip_high))
        v = np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)

    if gamma is not None and gamma > 0:
        v = np.power(v, gamma)  # gamma<1 boosts dim; gamma>1 compresses brights

    # --- NEW: median filter (great for salt/pepper) ---
    if (median_xy_px and median_xy_px > 0) or (median_z_slices and median_z_slices > 0):
        sz_z  = max(1, int(median_z_slices)) if median_z_slices else 1
        sz_xy = max(1, int(median_xy_px))    if median_xy_px    else 1
        v = ndi.median_filter(v, size=(sz_z, sz_xy, sz_xy))

    # --- NEW: grayscale opening per XY slice (erodes tiny bright dots, then restores shape) ---
    if open_xy_px and open_xy_px > 0:
        se = morphology.disk(int(open_xy_px))
        for z in range(v.shape[0]):
            v[z] = morphology.opening(v[z], footprint=se)

    # Mild Gaussian (kept last; can set sigma_xy=0 if median/opening already enough)
    v = ndi.gaussian_filter(v, sigma=(sigma_z, sigma_xy, sigma_xy))
    return v


def threshold_3d(
    vol,
    method="otsu",         # "nonzero" | "fixed" | "percentile" | "otsu" | "local" | "sauvola" | "hysteresis" | "recon"
    block_size=31,
    offset=0.0,
    sauvola_k=0.2,         # for "sauvola"
    hys_low_rel=0.5,       # for "hysteresis": low = hys_low_rel * high
    recon_low_rel=0.5,     # for "recon": low = recon_low_rel * high
    fixed_t=0,           # for "fixed": absolute threshold in image units
    percentile_p=99.5,     # for "percentile": percentile in [0,100]
    percentile_per_slice=False  # compute percentile per Z slice if vol is 3D
):
    """
    Threshold a 2D/3D image into a boolean mask using several methods.

    method:
      - "nonzero" (aliases: "present", ">0"): mask = vol > nonzero_eps
      - "fixed"   (aliases: "abs", "absolute"): mask = vol >= fixed_t
      - "percentile" (aliases: "perc", "quantile"): t = np.percentile(vol, percentile_p); mask = vol >= t
         * If percentile_per_slice=True and vol is 3D, compute t per Z slice.
      - "otsu"    (alias: "global"): global Otsu
      - "local": per-slice local threshold (Gaussian)
      - "sauvola": per-slice Sauvola
      - "hysteresis": per-slice hysteresis (global high via Otsu, low = rel*high)
      - "recon": 3D geodesic reconstruction (seed at high, flood through low)
    """

    # -------- fixed absolute threshold ----
    if method in {"fixed", "abs", "absolute"}:
        return vol > fixed_t

    # -------- percentile threshold --------
    if method in {"percentile", "perc", "quantile"}:
        p = float(percentile_p)
        if not (0.0 <= p <= 100.0):
            raise ValueError("percentile_p must be in [0, 100].")
        if vol.ndim == 3 and percentile_per_slice:
            mask = np.zeros_like(vol, dtype=bool)
            for z in range(vol.shape[0]):
                t = np.percentile(vol[z], p)
                mask[z] = vol[z] >= t
            return mask
        else:
            t = np.percentile(vol, p)
            return vol >= t

    # -------- global Otsu -----------------
    if method in {"otsu", "global"}:
        th = filters.threshold_otsu(vol)
        return vol > th

    # -------- local (per-slice) methods -------------
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

    # -------- hysteresis (per-slice) ----------------
    if method == "hysteresis":
        high = filters.threshold_otsu(vol)     # global high
        low  = high * float(hys_low_rel)
        if vol.ndim == 2:
            return apply_hysteresis_threshold(vol, low, high)
        mask = np.zeros_like(vol, dtype=bool)
        for z in range(vol.shape[0]):
            mask[z] = apply_hysteresis_threshold(vol[z], low, high)
        return mask

    # -------- 3D morphological reconstruction -------
    if method == "recon":
        high = filters.threshold_otsu(vol)
        low  = high * float(recon_low_rel)
        seed  = (vol > high).astype(np.uint8)
        allow = (vol > low ).astype(np.uint8)
        rec = reconstruction(seed, allow, method='dilation')
        return rec.astype(bool)

    raise ValueError("method must be one of: 'nonzero', 'fixed', 'percentile', 'otsu', 'local', 'sauvola', 'hysteresis', or 'recon'")


def fill_and_clean(mask, close_xy=3, min_size_vox=50, open_xy_px=0):
    filled = mask.copy()
    for z in range(filled.shape[0]):
        # NEW: binary opening to drop tiny dots/filaments
        if open_xy_px and open_xy_px > 0:
            se = morphology.disk(int(open_xy_px))
            filled[z] = morphology.binary_opening(filled[z], footprint=se)

        filled[z] = ndi.binary_fill_holes(filled[z])
        if close_xy > 0:
            se = morphology.disk(close_xy)
            filled[z] = morphology.binary_closing(filled[z], footprint=se)

    # Stronger speckle removal by size
    cleaned = morphology.remove_small_objects(filled, min_size=min_size_vox, connectivity=3)
    return cleaned


def watershed_split(mask, z_um, xy_um, min_peak_dist_um=30):
    sampling = (z_um, xy_um, xy_um)
    dt = ndi.distance_transform_edt(mask, sampling=sampling)
    # Estimate peak spacing in voxels
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


def v_um3_from_diam_um(d_um):
    return (math.pi/6.0) * (d_um**3)

def volume_filter(labels, z_um, xy_um, min_diam_um=40, max_diam_um=1000):
    vox_um3 = z_um * (xy_um**2)
    min_vox = int(round(v_um3_from_diam_um(min_diam_um) / vox_um3))
    max_vox = int(round(v_um3_from_diam_um(max_diam_um) / vox_um3))
    out = np.zeros_like(labels, dtype=np.int32)
    k = 1
    for lab in np.unique(labels):
        if lab == 0: 
            continue
        sz = np.count_nonzero(labels == lab)
        if min_vox <= sz <= max_vox:
            out[labels == lab] = k
            k += 1
    return out

def save_label_tiff(outdir, labels):
    out_path = os.path.join(outdir, f"{fbase}_mask.tif")
    tiff.imwrite(out_path, labels.astype(np.uint16))

def label_stats(labels, z_um, xy_um):
    # Compute per-label stats in physical units
    props = measure.regionprops(labels)
    rows = []
    vox_um3 = z_um * (xy_um**2)
    for p in props:
        vox = p.area
        vol_um3 = vox * vox_um3
        eq_d_um = ((6*vol_um3)/math.pi)**(1/3) if vol_um3>0 else 0.0
        zc, yc, xc = p.centroid
        rows.append({
            "label": int(p.label),
            "voxels": int(vox),
            "volume_um3": float(vol_um3),
            "equiv_diam_um": float(eq_d_um),
            "z_centroid": float(zc),
            "y_centroid": float(yc),
            "x_centroid": float(xc),
        })
    return pd.DataFrame(rows)

# Visualization helpers ---------------------------------------------------
def _rescale01(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99.8))
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)

def show_pipeline_slice(z, raw, pre, mask, cleaned, dt, markers, labels, labels_filt,
                        figsize=(18,10), alpha=0.35, random_seed=0,
                        outdir=outdir, dpi=300, show=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from skimage.color import label2rgb

    # For reproducible random colors
    if random_seed is not None:
        np.random.seed(random_seed)

    dt_slice   = dt[z] if dt is not None else np.zeros_like(raw[z])
    dt_disp    = _rescale01(dt_slice)
    raw_disp   = _rescale01(raw[z])
    pre_disp   = _rescale01(pre[z])
    mask_disp  = mask[z].astype(float)
    clean_disp = cleaned[z].astype(float)
    lab_slice  = labels[z]
    labf_slice = labels_filt[z]

    # Colored overlays for labels
    lab_rgb  = label2rgb(lab_slice,  image=pre_disp, bg_label=0, alpha=alpha, kind='overlay')
    labf_rgb = label2rgb(labf_slice, image=pre_disp, bg_label=0, alpha=alpha, kind='overlay')

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    ax = axes.ravel()

    ax[0].imshow(raw_disp, cmap='gray'); ax[0].set_title(f"Raw (z={z})"); ax[0].axis('off')
    ax[1].imshow(pre_disp, cmap='gray'); ax[1].set_title("Preprocessed"); ax[1].axis('off')
    ax[2].imshow(mask_disp, cmap='gray'); ax[2].set_title("Threshold mask"); ax[2].axis('off')
    ax[3].imshow(clean_disp, cmap='gray'); ax[3].set_title("Filled/cleaned"); ax[3].axis('off')
    ax[4].imshow(dt_disp, cmap='gray'); ax[4].set_title("Distance transform"); ax[4].axis('off')

    # markers overlay (2D indices)
    ax[5].imshow(pre_disp, cmap='gray'); ax[5].set_title("Markers (peaks)"); ax[5].axis('off')
    if markers is not None and np.any(markers):
        m2d = markers[z] if markers.ndim == 3 else markers
        yy, xx = np.nonzero(m2d)
        if yy.size:
            ax[5].plot(xx, yy, '.', markersize=3)  # default color

    # labels overlay (colored)
    ax[6].imshow(lab_rgb);  ax[6].set_title("Watershed labels (colored)");     ax[6].axis('off')
    b = find_boundaries(lab_slice, mode='inner');  ax[6].contour(b, levels=[0.5])

    # filtered labels overlay (colored)
    ax[7].imshow(labf_rgb); ax[7].set_title("Size-filtered labels (colored)"); ax[7].axis('off')
    bf = find_boundaries(labf_slice, mode='inner'); ax[7].contour(bf, levels=[0.5])

    import matplotlib.pyplot as plt
    plt.tight_layout()

    if outdir is not None:
        out_path = os.path.join(outdir, f"{fbase}_mask_slice.pdf")
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved QC figure to: {outdir}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig



def remove_xy_edge_touching(labels, margin_px=0, max_touch_px=0, relabel=True):
    """
    Keep objects that touch the XY border band by at most `max_touch_px` pixels (total across all Z).
    Remove objects that exceed this contact.

    labels:       3D int array (Z, Y, X)
    margin_px:    thickness (pixels) of the XY border band to test (0 => still tests the outermost edge)
    max_touch_px: allow up to this many touching pixels per object (0 => strict)
    relabel:      relabel sequentially from 1..N after removal
    """
    import numpy as np
    from skimage import segmentation

    if labels.ndim != 3:
        raise ValueError("labels must be 3D (Z, Y, X)")

    labmax = int(labels.max())
    if labmax == 0:
        return labels

    pad = max(1, int(margin_px))  # include at least the outermost edge

    # Build XY border band
    band = np.zeros_like(labels, dtype=bool)
    band[:, :pad, :]  = True   # Y top
    band[:, -pad:, :] = True   # Y bottom
    band[:, :, :pad]  = True   # X left
    band[:, :, -pad:] = True   # X right

    # Count touching pixels per label
    touch_count = np.bincount(labels[band].ravel(), minlength=labmax+1)

    # Keep labels with contact <= max_touch_px
    keep = np.ones(labmax+1, dtype=bool)
    keep[0] = False  # background
    for lab in range(1, labmax+1):
        keep[lab] = (touch_count[lab] <= int(max_touch_px))

    out = labels.copy()
    bad_ids = np.nonzero(~keep)[0]
    if bad_ids.size:
        out[np.isin(out, bad_ids)] = 0

    return segmentation.relabel_sequential(out)[0] if relabel else out


def remove_xy_edge_touching_um(labels, xy_um_per_px, margin_um=0.0, max_touch_um=0.0, relabel=True):
    """
    Same as above, but specify tolerances in micrometers.
    xy_um_per_px: XY voxel size (µm/px)
    margin_um:    thickness of border band to test (µm)
    max_touch_um: allow up to this many µm of contact (converted to pixels)
    """
    import math
    margin_px    = int(math.ceil(float(margin_um)    / float(xy_um_per_px))) if margin_um    > 0 else 0
    max_touch_px = int(math.ceil(float(max_touch_um) / float(xy_um_per_px))) if max_touch_um > 0 else 0
    return remove_xy_edge_touching(labels, margin_px=margin_px, max_touch_px=max_touch_px, relabel=relabel)

# ---------- Load both channels from ZCYX and combine for segmentation ----------

def load_channels_ZCYX(path, ch_dapi=0, ch_phall=1):
    """
    Load DAPI and phalloidin from a ZCYX stack and return as float32 (Z,Y,X).
    """
    vol = tiff.imread(path)
    if vol.ndim != 4 or not (vol.shape[1] <= 8 and vol.shape[2] > 8 and vol.shape[3] > 8):
        raise ValueError(f"Expected ZCYX (Z,C,Y,X). Got {vol.shape}")
    dapi = util.img_as_float32(vol[:, ch_dapi, :, :])      # (Z,Y,X)
    phal = util.img_as_float32(vol[:, ch_phall, :, :])     # (Z,Y,X)
    return dapi, phal

def combine_channels(pre_dapi, pre_phal, mode="max", w_dapi=1.0, w_phal=1.0):
    """
    Create a single 3D image to segment from the two preprocessed channels.
    mode: 'max' (robust), 'sum' (weighted), 'dapi', or 'phall'.
    """
    if mode == "dapi":
        return pre_dapi
    if mode == "phall":
        return pre_phal
    if mode == "sum":
        s = w_dapi*pre_dapi + w_phal*pre_phal
        # rescale per-volume for stability
        lo, hi = np.percentile(s, (1, 99.8))
        return np.clip((s - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
    # default: max-combine keeps strongest signal from either channel
    return np.maximum(pre_dapi, pre_phal).astype(np.float32)

def rgb_merge_slice(dapi_2d, phal_2d, mode="GB"):
    """
    Make a false-color RGB slice for QC.
    mode 'GB': Phalloidin=Green, DAPI=Blue (common).
    mode 'MB': Phalloidin=Magenta (R), DAPI=Blue.
    """
    d = _rescale01(dapi_2d)
    p = _rescale01(phal_2d)
    if mode == "MB":
        # Magenta (R) for phalloidin, Blue for DAPI
        return np.stack([p, np.zeros_like(p), d], axis=-1)
    # Default: Green for phalloidin, Blue for DAPI
    return np.stack([np.zeros_like(p), p, d], axis=-1)

from skimage.segmentation import find_boundaries, mark_boundaries

def show_overlay_slice(z, dapi, phal, labels=None, mask=None, color_mode="GB"):
    """
    QC: show merged RGB + optional mask/labels overlays for slice z.
    - labels: instance labels -> yellow boundaries over RGB
    - mask: binary mask -> cyan overlay
    """
    import matplotlib.pyplot as plt

    rgb = rgb_merge_slice(dapi[z], phal[z], mode=color_mode)

    ncols = 3 if (labels is not None or mask is not None) else 2
    fig, ax = plt.subplots(1, ncols, figsize=(5*ncols, 5))

    ax0 = ax[0] if ncols > 1 else ax
    ax0.imshow(_rescale01(dapi[z]), cmap="gray"); ax0.set_title(f"DAPI z={z}"); ax0.axis("off")
    ax1 = ax[1] if ncols > 1 else ax
    ax1.imshow(_rescale01(phal[z]), cmap="gray"); ax1.set_title("Phalloidin"); ax1.axis("off")

    if ncols == 3:
        vis = rgb.copy()
        if mask is not None:
            m = mask[z].astype(bool)
            # cyan mask overlay
            vis[m] = np.clip(vis[m] + np.array([0.0, 1.0, 1.0]), 0, 1)
        if labels is not None:
            b = find_boundaries(labels[z], mode="inner")
            # draw yellow boundaries
            vis[b] = np.array([1.0, 1.0, 0.0])
        ax[2].imshow(vis); ax[2].set_title("RGB + overlays"); ax[2].axis("off")

    plt.tight_layout(); plt.show()




#=============Run==============

files = sorted(
    [p for p in glob.glob(os.path.join(ROOT_DIR, "*.tif")) + glob.glob(os.path.join(ROOT_DIR, "*.tiff"))],
    key=lambda x: natsort.natsort_key(x)
)
print(f"Found {len(files)} TIFs.")

for fpath in files:
    fname = os.path.basename(fpath)
    fbase = fname[:-4]
    sample = sample_from_filename(fname, remove_from_file_name)
    print("\n=== Processing:", fname, "===")

    # Per-file output folder
    out_dir = outdir
    os.makedirs(out_dir, exist_ok=True)


    # ===for using all channels comment out if using only phalloidin and dapi===
    # # 1) Combine any channels you want straight from the file
    # pre_input = combine_channels_ZCYX(
    #     fpath,
    #     channels=[0, 1, 2, 3, 4],         # or [0,1,2,3], etc.
    #     mode="max",              # 'max' | 'sum' | 'mean'
    #     # weights=[1,0.6,1,0.2], # only if mode is 'sum' or 'mean'
    #     rescale=True
    # )  # -> (Z, Y, X)

    # # 2) Preprocess the combined volume once (same function you already use)
    # pre_combined = preprocess(
    #     pre_input,
    #     sigma_xy=sigma_xy, sigma_z=sigma_z,
    #     clip_low=clip_low, clip_high=clip_high,
    #     flatfield=True, bg_sigma_xy=bg_sigma_xy, bg_sigma_z=bg_sigma_z,
    #     per_slice_norm=True, gamma=gamma,
    #     median_xy_px=median_xy_px, median_z_slices=0, open_xy_px=open_xy_px_pre
    # )

    # ---------------- Example: run full segmentation using both channels ----------------
    # Inputs you already have:
    # fpath, out_path, z_spacing_um_per_px, xy_spacing_um_per_px
    # plus your chosen preprocessing / threshold / cleanup params

    # 1) Load both channels
    raw_dapi, raw_phal = load_channels_ZCYX(fpath, ch_dapi=0, ch_phall=1)

    # 2) Preprocess each (use your existing preprocess with your params)
    pre_dapi = preprocess(raw_dapi,
                          sigma_xy=sigma_xy, sigma_z=sigma_z,
                          clip_low=clip_low, clip_high=clip_high,
                          flatfield=True, bg_sigma_xy=bg_sigma_xy, bg_sigma_z=bg_sigma_z,
                          per_slice_norm=True, gamma=gamma,
                          median_xy_px=median_xy_px, median_z_slices=0, open_xy_px=open_xy_px_pre)

    pre_phal = preprocess(raw_phal,
                          sigma_xy=sigma_xy, sigma_z=sigma_z,
                          clip_low=clip_low, clip_high=clip_high,
                          flatfield=True, bg_sigma_xy=bg_sigma_xy, bg_sigma_z=bg_sigma_z,
                          per_slice_norm=True, gamma=gamma,
                          median_xy_px=median_xy_px, median_z_slices=0, open_xy_px=open_xy_px_pre)

    # 3) Combine to a single image to segment (robust default: 'max')
    pre_combined = combine_channels(pre_dapi, pre_phal, mode="max")  # or 'sum', 'dapi', 'phall'

    # 4) Threshold -> mask
    if method == "recon":
        mask = threshold_3d(pre_combined, method="recon", recon_low_rel=recon_low_rel)
    elif method == "hysteresis":
        mask = threshold_3d(pre_combined, method="hysteresis", hys_low_rel=hys_low_rel)
    elif method == "local":
        mask = threshold_3d(pre_combined, method="local", block_size=block_size, offset=offset)
    elif method == "otsu":
        mask = threshold_3d(pre_combined, method="otsu")
    elif method == "percentile":
        mask = threshold_3d(pre_combined, method="percentile", percentile_p=percentile_p)
    else:
        mask = threshold_3d(pre_combined, method="fixed", fixed_t = fixed_t)


    # 5) Clean mask
    mask_clean = fill_and_clean(mask, open_xy_px=open_xy_px_mask, close_xy=close_xy, min_size_vox=min_size_vox)
    print("mask_clean voxels:", int(mask_clean.sum()))
    print("dt max (um):", float(ndi.distance_transform_edt(mask_clean, sampling=(z_spacing_um_per_px, xy_spacing_um_per_px, xy_spacing_um_per_px)).max()))

    # (Optional) prune thin bridges before cleanup, if merging is a problem:
    #mask_clean = break_thin_bridges(mask_clean, z_um=z_spacing_um_per_px, xy_um=xy_spacing_um_per_px,min_half_width_um=min_half_width_um, redilate_um=redilate_um)

    # 6) Watershed split + size filter
    dt, markers, labels = watershed_split(mask_clean, z_um=z_spacing_um_per_px, xy_um=xy_spacing_um_per_px,
                                        min_peak_dist_um=min_peak_dist_um)

    labels_filt = volume_filter(labels, z_um=z_spacing_um_per_px, xy_um=xy_spacing_um_per_px,
                                min_diam_um=min_diam_um, max_diam_um=max_diam_um)

    # (Optional) drop objects touching XY border:
    labels_filt = remove_xy_edge_touching_um(labels_filt, xy_um_per_px=xy_spacing_um_per_px,
                                                margin_um=0, max_touch_um=100.0)

    # 7) Save & QC
    save_label_tiff(out_dir, labels_filt)

    # Pick a slice to view
    Z = pre_combined.shape[0]
    z_show = Z//2 if z_idx_for_qc is None else int(z_idx_for_qc)

    print("Labels before size filter:", int(labels.max()))
    print("Labels after size filter:", int( labels_filt.max()))

    pre_input = raw_dapi 

    show_pipeline_slice(z_show, pre_input, pre_combined, mask, mask_clean, dt, markers, labels, labels_filt,
                            figsize=(18,10), alpha=0.35, random_seed=0,
                            outdir=out_dir, dpi=300, show=False)