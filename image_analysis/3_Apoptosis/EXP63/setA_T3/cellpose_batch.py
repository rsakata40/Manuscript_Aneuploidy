#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------- Imports ---------
import os, glob, natsort, math, json, time
import numpy as np
import pandas as pd
import skimage.io as skio
from skimage.measure import label, regionprops
from cellpose import models, core, io

import matplotlib
matplotlib.use("Agg")           # headless backend
import matplotlib.pyplot as plt
plt.ioff()

def savefig_close(fig, outpath, dpi=300):
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# --------- USER CONFIG ---------
ROOT_DIR = '/ceph.groups/mshahbazi.grp/rsakata/EXP63/LIF2TIF/setA_T3'       # folder with .tif/.tiff
OUT_DIR = '/ceph.groups/mshahbazi.grp/rsakata/EXP63/output/setA_T3/1_cellpose'
mask_dir = "/ceph.groups/mshahbazi.grp/rsakata/EXP63/output/setA_T3/1_cellpose_mask"
out_dir = OUT_DIR

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Up to 5 channels (in the TIFF, AFTER reordering to Z,C,Y,X)
CHANNEL_ORDER = ["dapi", "GATA3", "GFP", "ZO1", "NANOG"]  # edit names/order to match your data
NUC_NAME      = "dapi"                                # nuclear channel name used for segmentation

# Voxel size (µm) and optional min volume (µm³) filter for objects
voxel_size_um = (2.0, 0.7633174, 0.7633174)     # (z_um, y_um, x_um) — edit!
minV_um3      = 250.0                 # set to None to skip size filtering

#forsegmentation
cellproblimit = 0
flowlimit = 0.4
manualD = None

# --------- Small helpers ---------
def reorder_tiff_to_zcxy(img):
    """Infer axes and reorder to (Z,C,Y,X)."""
    if img.ndim != 4:
        raise ValueError(f"Expected 4D TIF, got {img.shape}")
    shape = img.shape
    dim_map = {}
    for i, d in enumerate(shape):
        if d < 6:
            dim_map['C'] = i
        elif 7 < d < 500:
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

def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(ROOT_DIR, "*.tif")) + glob.glob(os.path.join(ROOT_DIR, "*.tiff")),
                key=lambda p: natsort.natsort_key(p))
    print(f"Found {len(files)} TIFs")

    # Slurm array support (one file per task)
    tid = int(os.getenv("SLURM_ARRAY_TASK_ID", "-1"))
    if tid >= 0:
        if tid < 0 or tid >= len(files):
            print(f"[array] task {tid} out of range (0..{len(files)-1}). Exiting.")
            return
        files = [files[tid]]
        print(f"[array] task {tid}: {files[0]}")

    for fpath in files:
        fname   = os.path.basename(fpath)
        sample  = sample_from_filename(fname)
        out_csv = os.path.join(OUT_DIR, f"{sample}_nucleus_intensity.csv")
        print(f"\n== {fname} → {sample}")

        #Skip if output for this sample already exists
        if os.path.exists(out_csv):         
            print(f"[skip] Found existing output: {out_csv}")
            continue

        # Load & reorder
        img_raw = skio.imread(fpath)
        #img     = reorder_tiff_to_zcxy(img_raw)          # (Z,C,Y,X)
        img = img_raw
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
            diameter=manualD, flow_threshold=flowlimit, cellprob_threshold=cellproblimit,
            min_size=minV_vox,  # min_size here is a post-filter inside Cellpose; we also filter ourselves below
            progress=False
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

        labeled_mask_16 = labeled_mask.astype('uint16')  # if <= 65535 labels
        io.save_masks(img, labeled_mask_16, flows, fpath,  savedir = mask_dir, save_flows=False, tif=True)

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

        # Save
        df_cells.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

        # ----- IMAGE: DAPI-only 7-slice mosaics -----
        zs = seven_slices(Z)

        # Normalize DAPI (nuclear) channel used for segmentation
        dapi_norm = percentile_norm(img[:, nuc_ch, :, :])

        # Try Phalloidin for gray background if available; else None
        phal_idx  = name2idx.get("phalloidin", None)
        phal_norm = percentile_norm(img[:, phal_idx, :, :]) if phal_idx is not None else None
        title_a   = "DAPI blue + Phalloidin yellow" if phal_norm is not None else "DAPI gray"

        # Figure/Grid with 2 rows (a,b)
        n_rows = 2
        fig = plt.figure(figsize=(21, 6))
        Y, X = dapi_norm.shape[-2:]  # in case Y,X aren't already defined

        # Row (a)
        for i, z in enumerate(zs):
            ax = plt.subplot(n_rows, 7, i + 1)
            if phal_norm is not None:
                rgb = np.zeros((Y, X, 3), float)
                rgb[..., :] = phal_norm[z][:, :, None]   # gray background from phalloidin
                rgb[..., 2] = dapi_norm[z]               # DAPI in blue
                ax.imshow(rgb, interpolation='none')
            else:
                ax.imshow(dapi_norm[z], cmap='gray', interpolation='none')
            ax.set_axis_off()
            ax.set_title(f"Z={z}")

        # Row (b): DAPI gray + DAPI masks
        rng = np.random.default_rng(0)
        col_table = rng.random((max(1, int(labeled_mask.max()) + 1), 3))
        col_table[0] = 0.0
        for i, z in enumerate(zs):
            ax = plt.subplot(n_rows, 7, 7 + i + 1)  # second row start
            ax.imshow(dapi_norm[z], cmap='gray', interpolation='none')
            sl = labeled_mask[z].astype(int)
            ax.imshow(col_table[sl], interpolation='none', alpha=0.55)
            ax.set_axis_off()

        # Row labels & save
        fig.text(0.02, 0.86, f"(a) {title_a}", rotation=90, va='center', ha='left')
        fig.text(0.02, 0.50, "(b) DAPI gray + DAPI masks", rotation=90, va='center', ha='left')

        plt.subplots_adjust(left=0.08, top=0.92, wspace=0.02, hspace=0.12)
        plt.suptitle(f"{sample} — 7-slice mosaics", y=0.98, fontsize=12)

        png_dapi = os.path.join(out_dir, f"{sample}_mosaic_Image1_a-b.png")
        savefig_close(fig, png_dapi)
        print("Saved:", png_dapi)

        import gc
        plt.close('all')
        gc.collect()

    print(f"\nAll done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
