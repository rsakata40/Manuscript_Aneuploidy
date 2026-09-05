#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------- Imports ---------
import os, glob, natsort
import numpy as np
import pandas as pd
import skimage.io as skio
from skimage.measure import label, regionprops
from cellpose import models, core

import matplotlib
matplotlib.use("Agg")           # headless backend
import matplotlib.pyplot as plt
plt.ioff()

def savefig_close(fig, outpath, dpi=300):
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# --------- USER CONFIG ---------
ROOT_DIR = '/ceph.groups/mshahbazi.grp/rsakata/EXP50/LIF2TIF'        # folder with .tif/.tiff
OUT_DIR = '/ceph.groups/mshahbazi.grp/rsakata/EXP50/cellpose_withstruc/cellpose_intensities'
out_dir = OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# Up to 5 channels (in the TIFF, AFTER reordering to Z,C,Y,X)
CHANNEL_ORDER = ["dapi", "phalloidin", "gfp", "rfp"]  # edit names/order to match your data
NUC_NAME      = "dapi"                                # nuclear channel name used for segmentation

# Voxel size (µm) and optional min volume (µm³) filter for objects
voxel_size_um = (2.0, 0.38, 0.38)     # (z_um, y_um, x_um) — edit!
minV_um3      = 800.0                 # set to None to skip size filtering

# Optional spheroid label maps (same shape as image (Z,Y,X))
SPH_LABELS_DIR = "/ceph.groups/mshahbazi.grp/rsakata/EXP50/cellpose_withstruc/mask"   # set to None to disable
SPH_SUFFIX     = "_mask.tif"                          # how spheroid label stacks are named

# --------- Small helpers ---------
def reorder_tiff_to_zcxy(img):
    """Infer axes and reorder to (Z,C,Y,X)."""
    if img.ndim != 4:
        raise ValueError(f"Expected 4D TIF, got {img.shape}")
    shape = img.shape
    dim_map = {}
    for i, d in enumerate(shape):
        if d < 10:
            dim_map['C'] = i
        elif 7 < d < 300:
            dim_map['Z'] = i
        else:
            if 'Y' not in dim_map: dim_map['Y'] = i
            else: dim_map['X'] = i
    if set(dim_map) != {'Z','C','Y','X'}:
        raise ValueError(f"Cannot infer Z/C/Y/X from shape {shape}, inferred {dim_map}")
    out = np.transpose(img, axes=[dim_map['Z'], dim_map['C'], dim_map['Y'], dim_map['X']])
    print(f"Reordered {shape} -> {out.shape} (Z,C,Y,X)")
    return out

def name_to_index_map(img, channel_order):
    C = img.shape[1]
    names = channel_order[:C]
    return {n: i for i, n in enumerate(names)}

def filter_small_objects_3d(masks, min_volume_vox):
    """Keep only objects with area >= min_volume_vox (voxels)."""
    lab0 = masks.astype(np.int32)
    if lab0.max() <= 1:
        lab0 = label(lab0 > 0, connectivity=3)
    out = np.zeros_like(lab0, dtype=np.uint16)
    nid = 1
    for p in regionprops(lab0):
        if p.area >= min_volume_vox:
            out[lab0 == p.label] = nid
            nid += 1
    return out

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

def find_spheroid_labels_path(sample, fpath):
    """Return path to spheroid labels if present; None otherwise."""
    if SPH_LABELS_DIR is None:
        return None
    cand1 = os.path.join(SPH_LABELS_DIR, f"{sample}{SPH_SUFFIX}")
    if os.path.exists(cand1):
        return cand1
    root, _ = os.path.splitext(os.path.basename(fpath))
    cand2 = os.path.join(SPH_LABELS_DIR, root + SPH_SUFFIX)
    return cand2 if os.path.exists(cand2) else None

def map_cells_to_spheroids(cell_lab, sph_lab):
    """Largest-overlap mapping per cell → spheroid (or 0)."""
    if cell_lab.shape != sph_lab.shape:
        raise ValueError(f"Shape mismatch: cells {cell_lab.shape} vs sph {sph_lab.shape}")
    c = cell_lab.ravel().astype(np.int64, copy=False)
    s = sph_lab.ravel().astype(np.int64, copy=False)
    cmax = int(c.max())
    if cmax == 0:
        return pd.DataFrame(columns=["cell_label","spheroid_label","overlap_vox","frac_of_cell","frac_of_spheroid"])
    m = (c > 0)
    pairs = np.stack([c[m], s[m]], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    df_pairs = pd.DataFrame({"cell_label": uniq[:,0], "spheroid_label": uniq[:,1], "overlap_vox": counts})
    # pick best positive spheroid if any; else best against 0
    df_pos = df_pairs[df_pairs.spheroid_label > 0]
    best_pos = df_pos.loc[df_pos.groupby("cell_label")["overlap_vox"].idxmax()] if not df_pos.empty else df_pos
    cells_all = np.arange(1, cmax+1)
    missing = np.setdiff1d(cells_all, best_pos["cell_label"].to_numpy(dtype=int, copy=False))
    if missing.size:
        df_s0 = df_pairs[(df_pairs.spheroid_label == 0) & (df_pairs.cell_label.isin(missing))]
        if not df_s0.empty:
            best_s0 = df_s0.loc[df_s0.groupby("cell_label")["overlap_vox"].idxmax()]
        else:
            best_s0 = pd.DataFrame({"cell_label": missing, "spheroid_label": 0, "overlap_vox": 0})
        df_best = pd.concat([best_pos, best_s0], ignore_index=True)
    else:
        df_best = best_pos.copy()
    # fractions
    cell_vox = np.bincount(c, minlength=cmax+1).astype(float)
    sph_vox  = np.bincount(s).astype(float)
    ov = df_best["overlap_vox"].to_numpy(float)
    cv = cell_vox[df_best["cell_label"].to_numpy(int)]
    sv = sph_vox[np.minimum(df_best["spheroid_label"].to_numpy(int), len(sph_vox)-1)]
    frac_cell = np.divide(ov, np.maximum(cv, 1.0), out=np.zeros_like(ov), where=cv>0)
    frac_sph  = np.zeros_like(ov)
    pos = df_best["spheroid_label"].to_numpy(int) > 0
    np.divide(ov[pos], np.maximum(sv[pos], 1.0), out=frac_sph[pos], where=sv[pos]>0)
    df_best = df_best.assign(frac_of_cell=frac_cell, frac_of_spheroid=frac_sph).sort_values("cell_label").reset_index(drop=True)
    return df_best

def seven_slices(Z):
    return np.linspace(0, Z-1, 7, dtype=int)

def percentile_norm(vol, p1=1, p99=99):
    """Normalize a (Z,Y,X) volume to 0..1 using global percentiles."""
    lo, hi = np.percentile(vol, (p1, p99))
    scale = max(hi - lo, 1e-6)
    return np.clip((vol - lo) / scale, 0, 1)

# --------- Model (once) ---------
model = models.CellposeModel(gpu=True)
print("GPU available to Cellpose:", core.use_gpu())

# --------- Batch ---------
files = sorted(glob.glob(os.path.join(ROOT_DIR, "*.tif")) + glob.glob(os.path.join(ROOT_DIR, "*.tiff")),
               key=lambda p: natsort.natsort_key(p))
print(f"Found {len(files)} TIFs")

for fpath in files:
    fname   = os.path.basename(fpath)
    sample  = sample_from_filename(fname)
    out_csv = os.path.join(OUT_DIR, f"{sample}_cells_with_spheroid.csv")
    print(f"\n== {fname} → {sample}")

    # Load & reorder
    img_raw = skio.imread(fpath)
    img     = reorder_tiff_to_zcxy(img_raw)          # (Z,C,Y,X)
    Z, C, Y, X = img.shape

    # Channel map
    name2idx = name_to_index_map(img, CHANNEL_ORDER)
    if NUC_NAME not in name2idx:
        raise ValueError(f"NUC_NAME='{NUC_NAME}' not found in CHANNEL_ORDER={CHANNEL_ORDER} for file with {C} channels.")
    nuc_ch = name2idx[NUC_NAME]

    # Segment on nuclear channel only (single-channel input to v4)
    img_cp = img[:, [nuc_ch], :, :]
    # optional physical -> vox cutoff
    if (voxel_size_um is not None) and (minV_um3 is not None):
        z_um, y_um, x_um = map(float, voxel_size_um)
        voxel_um3 = z_um * y_um * x_um
        minV_vox  = float(minV_um3) / voxel_um3
    else:
        minV_vox  = 0.0

    eval_kwargs = dict(
        do_3D=True, z_axis=0, channel_axis=1,
        diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
        min_size=minV_vox, progress=False
    )
    # anisotropy if known
    if voxel_size_um is not None:
        z_um, y_um, x_um = map(float, voxel_size_um)
        eval_kwargs["anisotropy"] = z_um / y_um if y_um > 0 else None

    masks, flows, styles = model.eval(img_cp, **eval_kwargs)   # (Z,Y,X) labels
    masks_pref = masks.astype(np.uint16)

    # Extra size filter (keeps labels sequential)
    masks_filt = filter_small_objects_3d(masks_pref, minV_vox) if minV_vox > 0 else masks_pref
    labeled_mask = masks_filt if masks_filt.max() > 1 else label(masks_filt > 0, connectivity=3)
    nobj = int(labeled_mask.max())
    print(f"Objects: {nobj}")

    # Per-object stats: volume, centroid, and mean intensity for each named channel
    rows = []
    vox_um3 = None
    if voxel_size_um is not None:
        z_um, y_um, x_um = map(float, voxel_size_um)
        vox_um3 = z_um * y_um * x_um

    props_all = regionprops(labeled_mask)
    for p in props_all:
        lab = int(p.label)
        cz, cy, cx = (float(v) for v in p.centroid)
        row = dict(
            file_name=fname,
            sample=sample,
            ObjectID=lab,
            Volume_voxels=int(p.area),
            CentroidZ_px=cz, CentroidY_px=cy, CentroidX_px=cx
        )
        if vox_um3:
            row["Volume_um3"]  = float(p.area) * vox_um3
            row["CentroidZ_um"] = cz * z_um
            row["CentroidY_um"] = cy * y_um
            row["CentroidX_um"] = cx * x_um
        # means per channel name
        for ch_name, ch_idx in name2idx.items():
            vals = img[:, ch_idx, :, :][labeled_mask == lab]
            row[f"Mean_{ch_name}"] = float(vals.mean()) if vals.size else np.nan
        rows.append(row)

    df_cells = pd.DataFrame(rows)

    # Merge with spheroid labels (if available)
    sph_path = find_spheroid_labels_path(sample, fpath)
    if sph_path is not None and os.path.exists(sph_path):
        sph_lab = skio.imread(sph_path)
        if sph_lab.ndim != 3:
            raise ValueError(f"Spheroid labels must be (Z,Y,X); got {sph_lab.shape} at {sph_path}")
        if sph_lab.shape != labeled_mask.shape:
            raise ValueError(f"Shape mismatch: cells {labeled_mask.shape} vs spheroids {sph_lab.shape}")
        df_map = map_cells_to_spheroids(labeled_mask, sph_lab)  # cell_label, spheroid_label, overlap_vox, frac_of_cell, frac_of_spheroid
        df_map = df_map.rename(columns={"cell_label": "ObjectID"})
        df_cells = df_cells.merge(df_map, on="ObjectID", how="left")
    else:
        print("No spheroid map for this sample (skipping merge).")

    # Save
    df_cells.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # ----- IMAGE: DAPI-only 7-slice mosaic (a) + (b) -----
    zs = seven_slices(Z)

    # Normalize DAPI (nuclear) channel used for segmentation
    dapi_norm = percentile_norm(img[:, nuc_ch, :, :])

    # Try Phalloidin for gray background if available; else None
    phal_idx  = name2idx.get("phalloidin", None)
    phal_norm = percentile_norm(img[:, phal_idx, :, :]) if phal_idx is not None else None
    title_a   = "DAPI blue + Phalloidin yellow" if phal_norm is not None else "DAPI gray"

    fig = plt.figure(figsize=(21, 6))

    # Row (a)
    for i, z in enumerate(zs):
        ax = plt.subplot(2, 7, i+1)
        if phal_norm is not None:
            rgb = np.zeros((Y, X, 3), float)
            rgb[..., :] = phal_norm[z][:, :, None]   # gray background
            rgb[..., 2] = dapi_norm[z]               # DAPI in blue
            ax.imshow(rgb, interpolation='none')
        else:
            ax.imshow(dapi_norm[z], cmap='gray', interpolation='none')
        ax.set_axis_off()
        ax.set_title(f"Z={z}")

    # Row (b): DAPI gray + DAPI masks (pre-filter)
    rng = np.random.default_rng(0)
    col_table = rng.random((max(1, int(masks_pref.max())+1), 3))
    col_table[0] = 0.0
    for i, z in enumerate(zs):
        ax = plt.subplot(2, 7, 7+i+1)
        ax.imshow(dapi_norm[z], cmap='gray', interpolation='none')
        sl = masks_pref[z].astype(int)
        ax.imshow(col_table[sl], interpolation='none', alpha=0.55)
        ax.set_axis_off()

    # Row labels & save
    fig.text(0.02, 0.73, f"(a) {title_a}", rotation=90, va='center', ha='left')
    fig.text(0.02, 0.25, "(b) DAPI gray + DAPI masks", rotation=90, va='center', ha='left')
    plt.subplots_adjust(left=0.08, top=0.92, wspace=0.02, hspace=0.12)
    plt.suptitle(f"{sample} — DAPI-only 7-slice mosaics", y=0.98, fontsize=12)

    png_dapi = os.path.join(out_dir, f"{sample}_mosaic_DAPImask_a-b.png")
    savefig_close(fig, png_dapi)
    print("Saved:", png_dapi)


    # ----- IMAGE 2: Volume histogram (pre-filter) in µm³ with minV_um3 line -----
    props_pref = regionprops(label(masks_pref > 0, connectivity=2))
    vols_vox   = np.array([p.area for p in props_pref], dtype=float)

    if vols_vox.size:
        if voxel_size_um is not None:
            z_um, y_um, x_um = map(float, voxel_size_um)
            voxel_um3 = z_um * y_um * x_um
            vols_um3 = vols_vox * voxel_um3
            xmax = max(float(np.nanmax(vols_um3)), float(minV_um3), 1.0)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(vols_um3, bins=60, range=(100, 20000))
            ax.axvline(minV_um3, linewidth=1)
            ax.set_xlim(100, 20000); ax.set_ylim(bottom=0)
            ax.set_xlabel("Object volume (µm³)")
            ax.set_ylabel("Count")
            ax.set_title("Volume distribution (pre-filter)")
            fig.tight_layout()
            png3 = os.path.join(out_dir, f"{sample}_hist_volume_prefilter_minV{int(minV_um3)}um3.png")
            savefig_close(fig, png3)
            print("Saved:", png3)

            minV_vox = float(minV_um3) / voxel_um3
            print(f"Minimum volume threshold: {minV_um3:.1f} µm³  (~{minV_vox:.1f} vox)")
            minV = int(round(minV_vox))
        else:
            xmax = max(float(np.nanmax(vols_vox)), 1.0)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(vols_vox, bins=60, range=(0, xmax))
            ax.set_xlim(0, xmax); ax.set_ylim(bottom=0)
            ax.set_xlabel("Object volume (voxels)")
            ax.set_ylabel("Count")
            ax.set_title("Volume distribution (pre-filter)")
            fig.tight_layout()
            png3 = os.path.join(out_dir, f"{sample}_hist_volume_prefilter_vox.png")
            savefig_close(fig, png3)
            print("Saved:", png3)
    else:
        print("No objects for volume histogram (pre-filter).")
    import gc
    plt.close('all')
    gc.collect()
