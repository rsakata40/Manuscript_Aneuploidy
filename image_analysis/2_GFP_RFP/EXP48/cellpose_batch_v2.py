#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### analysis of GFP/RFP positive cells and link them to blastoids object mask(if present)

# === Batch Cellpose 3D pipeline over a folder of TIFs ===
import os, glob, natsort, math, json, time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.measure import label, regionprops
from skimage.measure import label as sklabel, regionprops as skprops
from cellpose import models, core, plot



matplotlib.use("Agg")        # belt-and-suspenders
plt.ioff()                   # no interactive windows

# --------- USER CONFIG ---------
ROOT_DIR = '/ceph.groups/mshahbazi.grp/rsakata/EXP48/LIF2TIF'    # <— change
outdir = '/ceph.groups/mshahbazi.grp/rsakata/EXP48/cellpose/cellpose_intensities'
remove_from_file_name = None  # <— change if needed
minV_um3 = 800.0    # µm³

zum = 1.9998570 #the um/pixel ratio in Z
xyum = 0.7583174 #the um/pixel ratio in Z
voxel_size_um  = (zum, xyum, xyum)

# Gate settings (RFP >= m * GFP → RFP+, else GFP+)
#pos_thresholds = (50,40) # set to None if using slope
pos_thresholds = None
slope_m = 1.0
rfp_min = 25
gfp_min = 25

# Channel indices in the reordered (Z, C, Y, X) data
CH_DAPI = 0
CH_PHAL = 1
CH_GFP  = 2
CH_RFP  = 3

# where the spheroid label TIFs live and how they’re named ---
SPH_LABELS_DIR = None #"/ceph.groups/mshahbazi.grp/rsakata/EXP48/cellpose/mask"    # set this
SPH_SUFFIX = None               # e.g. "<sample>_spheroids.tif" in SPH_LABELS_DIR


#original
masks_h5 = 0
nuclabel = 1
nucchannel = 0      # DAPI (0-based after reordering to ZCXY)
optionalchannel = 1    # phalloidin
slice2check = 50
minsize2del = 10
c2d = 0
cellproblimit = 0
flowlimit = 0.4
manualD = None

# --------- HELPERS ---------

def find_spheroid_labels_path(sample, fpath):
    # If no labels directory is configured, don't search anywhere.
    if SPH_LABELS_DIR is None:
        return None

    cand1 = os.path.join(SPH_LABELS_DIR, f"{sample}{SPH_SUFFIX}")
    if os.path.exists(cand1):
        return cand1

    root, _ = os.path.splitext(os.path.basename(fpath))
    cand2 = os.path.join(SPH_LABELS_DIR, root + SPH_SUFFIX)
    return cand2 if os.path.exists(cand2) else None


def map_cells_to_spheroids(cell_lab, sph_lab):
    """
    Map each cell object (label > 0) in cell_lab to the spheroid (label >= 0) in sph_lab
    with which it has the largest voxel overlap.

    Returns a DataFrame with columns:
      cell_label, spheroid_label, overlap_vox, frac_of_cell, frac_of_spheroid
    """
    # --- checks ---
    if cell_lab.shape != sph_lab.shape:
        raise ValueError(f"Shape mismatch: cell_lab {cell_lab.shape} vs sph_lab {sph_lab.shape}")
    if cell_lab.dtype.kind not in "iu" or sph_lab.dtype.kind not in "iu":
        raise ValueError("Labels must be integer dtype.")
    if np.any(cell_lab < 0) or np.any(sph_lab < 0):
        raise ValueError("Labels must be non-negative (0 = background).")

    c = cell_lab.ravel().astype(np.int64, copy=False)
    s = sph_lab.ravel().astype(np.int64, copy=False)

    cmax, smax = int(c.max()), int(s.max())
    if cmax == 0:
        # no cells
        return pd.DataFrame(columns=["cell_label","spheroid_label","overlap_vox","frac_of_cell","frac_of_spheroid"])

    # voxel counts per label (include 0 so indexing works)
    cell_vox = np.bincount(c, minlength=cmax+1)
    sph_vox  = np.bincount(s, minlength=smax+1 if smax > 0 else 1)

    # consider only voxels where cell > 0 (cell background irrelevant)
    m = (c > 0)
    if not np.any(m):
        return pd.DataFrame(columns=["cell_label","spheroid_label","overlap_vox","frac_of_cell","frac_of_spheroid"])

    # Safer (uses one 1D code vector)
    cm = c[m].astype(np.int64, copy=False)
    sm = s[m].astype(np.int64, copy=False)
    base = int(smax) + 1
    code = cm * base + sm                       # 1 vector instead of (N,2)
    uniq, counts = np.unique(code, return_counts=True)
    cc = (uniq // base).astype(np.int64, copy=False)
    ss = (uniq %  base).astype(np.int64, copy=False)
    vv = counts

    # Build per-(cell,sph) rows and select best spheroid (max overlap) per cell, preferring sph>0 when tied
    df_pairs = pd.DataFrame({"cell_label": cc, "spheroid_label": ss, "overlap_vox": vv})

    # Split into positive-spheroid and spheroid=0 rows
    df_pos = df_pairs[df_pairs.spheroid_label > 0]
    # For cells that have at least one positive spheroid, take argmax there
    if not df_pos.empty:
        idx_max_pos = df_pos.groupby("cell_label")["overlap_vox"].idxmax()
        df_best_pos = df_pos.loc[idx_max_pos]
    else:
        df_best_pos = pd.DataFrame(columns=df_pairs.columns)

    # Cells that never touched a positive spheroid → pick their (cell, sph=0) overlap if present, else 0
    cells_all = np.arange(1, cmax+1, dtype=int)
    cells_with_pos = df_best_pos["cell_label"].to_numpy(dtype=int, copy=False)
    cells_missing = np.setdiff1d(cells_all, cells_with_pos, assume_unique=False)

    if cells_missing.size:
        # find overlaps against sph=0 for these cells
        df_s0 = df_pairs[(df_pairs["spheroid_label"] == 0) & (df_pairs["cell_label"].isin(cells_missing))]
        if not df_s0.empty:
            idx_max_s0 = df_s0.groupby("cell_label")["overlap_vox"].idxmax()
            df_best_s0 = df_s0.loc[idx_max_s0]
        else:
            # cells truly with no recorded overlap (should be rare unless masked)
            df_best_s0 = pd.DataFrame({
                "cell_label": cells_missing,
                "spheroid_label": np.zeros_like(cells_missing, dtype=int),
                "overlap_vox": np.zeros_like(cells_missing, dtype=int),
            })
        df_best = pd.concat([df_best_pos, df_best_s0], ignore_index=True)
    else:
        df_best = df_best_pos.copy()

    # Fractions
    # vectorized lookups into bincount arrays
    ov = df_best["overlap_vox"].to_numpy(dtype=float, copy=False)
    cv = cell_vox[df_best["cell_label"].to_numpy(dtype=int, copy=False)].astype(float)
    sv = sph_vox[np.minimum(df_best["spheroid_label"].to_numpy(dtype=int, copy=False), smax)].astype(float)

    frac_of_cell = np.divide(ov, np.maximum(cv, 1.0), out=np.zeros_like(ov, dtype=float), where=cv>0)
    # only defined for sph>0; set to 0 for sph=0
    sph_positive = df_best["spheroid_label"].to_numpy(copy=False) > 0
    frac_of_spheroid = np.zeros_like(ov, dtype=float)
    np.divide(ov[sph_positive], np.maximum(sv[sph_positive], 1.0),
              out=frac_of_spheroid[sph_positive], where=sv[sph_positive]>0)

    df_best = df_best.assign(
        frac_of_cell=frac_of_cell,
        frac_of_spheroid=frac_of_spheroid
    ).sort_values("cell_label").reset_index(drop=True)

    return df_best


def spheroid_stats_simple(labels_sph, z_um, xy_um):
    """Volume & centroid per spheroid."""
    vox_um3 = float(z_um) * (float(xy_um)**2)
    rows = []
    for p in regionprops(labels_sph):
        rows.append({
            "spheroid_label": int(p.label),
            "voxels": int(p.area),
            "volume_um3": float(p.area) * vox_um3,
            "z_centroid": float(p.centroid[0]),
            "y_centroid": float(p.centroid[1]),
            "x_centroid": float(p.centroid[2]),
        })
    return pd.DataFrame(rows)

# def reorder_tiff_to_zcxy(img):
#     """Infer axes and reorder to (Z, C, Y, X)."""
#     shape = img.shape
#     if img.ndim != 4:
#         raise ValueError(f"Expected 4D TIF, got {shape}")
#     dim_map = {}
#     for i, d in enumerate(shape):
#         if d < 10:
#             dim_map['C'] = i
#         elif 5 < d < 300:
#             dim_map['Z'] = i
#         else:
#             if 'Y' not in dim_map: dim_map['Y'] = i
#             else: dim_map['X'] = i
#     if set(dim_map) != {'Z','C','Y','X'}:
#         raise ValueError(f"Cannot infer Z/C/Y/X from shape {shape}, inferred {dim_map}")
#     out = np.transpose(img, axes=[dim_map['Z'], dim_map['C'], dim_map['Y'], dim_map['X']])
#     print(f"Original shape: {shape} → Reordered to (Z, C, Y, X): {out.shape}")
#     return out

def reorder_tiff_to_zcxy(img):
    """
    Reorder a 4D array from ZYXC to ZCYX.
    """
    if img.ndim != 4:
        raise ValueError(f"Expected a 4D array shaped (Z, Y, X, C); got {img.shape}")
    out = np.transpose(img, (0, 3, 1, 2))
    shape = img.shape
    print(f"Original shape: {shape}" )
    return out


def filter_small_objects_3d(masks, min_volume_vox):
    """Relabel and keep only objects with area >= min_volume_vox (voxels)."""
    labeled0 = masks.astype(np.int32)
    if labeled0.max() <= 1:
        labeled0 = label(labeled0 > 0, connectivity=3)
    props = regionprops(labeled0)
    out = np.zeros_like(labeled0, dtype=np.uint16)
    nid = 1
    for p in props:
        if p.area >= min_volume_vox:
            out[labeled0 == p.label] = nid
            nid += 1
    return out

def percentile_norm(vol, p1=1, p99=99):
    """Normalize a (Z,Y,X) volume to 0..1 using global percentiles."""
    lo, hi = np.percentile(vol, (p1, p99))
    scale = max(hi - lo, 1e-6)
    return np.clip((vol - lo) / scale, 0, 1)

def seven_slices(Z):
    return np.linspace(0, Z-1, 7, dtype=int)

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

def add_file_columns(df, short_name, sample_name):
    df = df.copy()
    df.insert(0, "file_name", short_name)
    df.insert(1, "sample", sample_name)
    return df

def centroid_df_from_mask(labeled_mask):
    rows = []
    for p in regionprops(labeled_mask):
        z, y, x = (float(v) for v in p.centroid)  # (Z, Y, X) in pixel indices
        row = {
            "ObjectID": int(p.label),
            "CentroidZ_px": z,
            "CentroidY_px": y,
            "CentroidX_px": x,
        }
        rows.append(row)
    return pd.DataFrame(rows)

def savefig_close(fig, outpath, dpi=300, bbox_inches="tight"):
    fig.savefig(outpath, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)

# --------- MODEL (create once) ---------
model = models.CellposeModel(gpu=True)  # v4 ignores model_type and channels=
print(">>> GPU:", core.use_gpu())

# Anisotropy (optional but helpful). If not provided, try to derive from voxel_size_um
if 'voxel_size_um' in globals() and voxel_size_um is not None:
    zum, xyum, _xyum2 = voxel_size_um
    try:
        anisotropy = float(zum) / float(xyum)
    except Exception:
        anisotropy = None
else:
    anisotropy = None  # set later if you like

# --------- GLOB FILES ---------


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
        fname = os.path.basename(fpath)
        sample = sample_from_filename(fname, remove_from_file_name)
        print("\n=== Processing:", fname, "→ sample:", sample, "===")

        # Per-file output folder
        out_dir = outdir
        os.makedirs(out_dir, exist_ok=True)

        # Skip if output for this sample already exists
        out_csv = os.path.join(out_dir, f"{sample}_cells_with_spheroid.csv")
        if os.path.exists(out_csv):         
            print(f"[skip] Found existing output: {out_csv}")
            continue

        # ----- LOAD -----
        img_raw = skio.imread(fpath)
        img = reorder_tiff_to_zcxy(img_raw)  # (Z,C,Y,X)
        Z, C, Y, X = img.shape
        z_um, y_um, x_um = voxel_size_um
        voxel_um3 = float(z_um) * float(y_um) * float(x_um)
        minV = minV_um3 / voxel_um3 

        # ----- SEGMENT (3D) -----
        # Build 2-channel view for CPSAM: [Phalloidin, DAPI]
        img_cp = img[:, [CH_PHAL, CH_DAPI], :, :]
        eval_kwargs = dict(
            do_3D=True, z_axis=0, channel_axis=1,
            diameter=manualD, flow_threshold=flowlimit, cellprob_threshold=cellproblimit,
            min_size=minV,  # min_size here is a post-filter inside Cellpose; we also filter ourselves below
            progress=False
        )
        if anisotropy is not None:
            eval_kwargs["anisotropy"] = anisotropy

        masks, flows, styles = model.eval(img_cp, **eval_kwargs)  # masks: (Z,Y,X), labeled

        # Keep a copy before our filter for histogram (pre-filter volumes)
        masks_prefilter = masks.astype(np.uint16)

        # ----- FILTER SMALL OBJECTS ----- 
        masks_filtered = filter_small_objects_3d(masks_prefilter, min_volume_vox=minV)
        n_before, n_after = int(masks_prefilter.max()), int(masks_filtered.max())
        print(f"Objects: before={n_before}, after={n_after} (minV={minV} vox)")

        # ----- PER-CELL MEANS & GATING (thresholds preferred) -----
        labeled_mask = masks_filtered if masks_filtered.max() > 1 else label(masks_filtered > 0, connectivity=3)

        props_g = regionprops(labeled_mask, intensity_image=img[:, CH_GFP, :, :])
        props_r = regionprops(labeled_mask, intensity_image=img[:, CH_RFP, :, :])

        gfp_means = np.array([p.mean_intensity for p in props_g], float)
        rfp_means = np.array([p.mean_intensity for p in props_r], float)

        #valid = (gfp_means + rfp_means) >= float(min_total_for_gate)

        # Use (GFP_thresh, RFP_thresh) if provided; otherwise fall back to slope gate
        _use_thresh = (
            isinstance(pos_thresholds, (tuple, list)) and len(pos_thresholds) == 2
            and np.isfinite(pos_thresholds[0]) and np.isfinite(pos_thresholds[1])
        )

        if _use_thresh:
            GFP_thresh, RFP_thresh = float(pos_thresholds[0]), float(pos_thresholds[1])

            # independent positivity by channel, then restricted to 'valid'
            Gpos = (gfp_means >= GFP_thresh) 
            Rpos = (rfp_means >= RFP_thresh)

            gate_desc = f"thresholds GFP≥{GFP_thresh:g}, RFP≥{RFP_thresh:g}"
        else:
            # fallback: diagonal slope gate (RFP ≥ m * GFP → RFP+, else GFP+), with 'valid' filter
            Rpos = (rfp_means >= float(slope_m) * gfp_means) & (rfp_means >= rfp_min)
            Gpos = (~Rpos)  & (gfp_means >= gfp_min)
            gate_desc = f"slope gate m={slope_m}"

        labels_vec = np.arange(1, labeled_mask.max()+1)


        # ----- SUMMARY CSV (filtered) -----
        if 'extract_object_properties' in globals():
            print("using 'extract_object_properties' in globals()")
            out_csv = os.path.join(out_dir, f"{sample}_summary_table_filtered.csv")
            df = extract_object_properties(
                masks_filtered, img, out_csv,
                voxel_size_um=voxel_size_um if 'voxel_size_um' in globals() else None,
                pos_channels=(CH_GFP, CH_RFP),
                pos_thresholds=pos_thresholds,                 # use ratio gate internally if you patched it earlier
                ratio_gate_slope=slope_m,
                ratio_gate_min_total=min_total_for_gate
            )
            # ---- NEW: append centroid columns ----
            # Use existing labeled_mask if present; otherwise create it from masks_filtered
            if 'labeled_mask' in globals() and labeled_mask is not None:
                _labeled = labeled_mask
            else:
                _labeled = label((masks_filtered > 0).astype(np.uint8))
            cent_df = centroid_df_from_mask(_labeled)
            # Expect df to have an ObjectID column; merge on that
            if 'ObjectID' not in df.columns:
                # Try to infer: if there is a 'label' column, rename it
                if 'label' in df.columns:
                    df = df.rename(columns={'label': 'ObjectID'})
                else:
                    raise KeyError("Could not find ObjectID column to merge centroid positions into.")
            df = df.merge(cent_df, on='ObjectID', how='left')

            # Append file/sample columns and rewrite
            df = add_file_columns(df, fname, sample)
            df.to_csv(out_csv, index=False)
        else:
            # Lightweight fallback (volumes + means + ratio gate + centroids)
            props_all = regionprops(labeled_mask)
            rows = []
            vox_um3 = None
            if 'voxel_size_um' in globals() and voxel_size_um is not None:
                z_um, y_um, x_um = voxel_size_um
                vox_um3 = float(z_um)*float(y_um)*float(x_um)
            for p in props_all:
                lab_id = int(p.label)
                # centroid in pixel indices (Z,Y,X)
                cz, cy, cx = (float(v) for v in p.centroid)
                row = dict(
                    ObjectID=lab_id,
                    Volume_voxels=p.area,
                    MeanIntensity_C0=float(np.mean(img[:, 0, :, :][labeled_mask == lab_id])),
                    MeanIntensity_C1=float(np.mean(img[:, 1, :, :][labeled_mask == lab_id])),
                    MeanIntensity_C2=float(gfp_means[lab_id-1] if lab_id-1 < len(gfp_means) else np.nan),
                    MeanIntensity_C3=float(rfp_means[lab_id-1] if lab_id-1 < len(rfp_means) else np.nan),
                    GFP_pos=bool(Gpos[lab_id-1]) if lab_id-1 < len(Gpos) else False,
                    RFP_pos=bool(Rpos[lab_id-1]) if lab_id-1 < len(Rpos) else False,
                    PosClass=("double+" if (lab_id-1 < len(Gpos) and Gpos[lab_id-1]) and (lab_id-1 < len(Rpos) and Rpos[lab_id-1]) else
                            "RFP+" if (lab_id-1 < len(Rpos) and Rpos[lab_id-1]) else
                            "GFP+" if (lab_id-1 < len(Gpos) and Gpos[lab_id-1]) else "neg"),
                    # ---- NEW: centroid in pixels ----
                    CentroidZ_px=cz,
                    CentroidY_px=cy,
                    CentroidX_px=cx,
                )
                if vox_um3:
                    row["Volume_um3"] = float(p.area)*vox_um3
                    # ---- NEW: centroid in microns ----
                    row["CentroidZ_um"] = cz * float(z_um)
                    row["CentroidY_um"] = cy * float(y_um)
                    row["CentroidX_um"] = cx * float(x_um)
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False)

        # --- load spheroid labels for this sample ---
        sph_path = find_spheroid_labels_path(sample, fpath)
        if sph_path is None:
            print("No spheroid labels found for this sample; skipping cell↔spheroid mapping.")
        else:
            sph_lab = skio.imread(sph_path)
            if sph_lab.ndim != 3:
                raise ValueError(f"Spheroid labels must be (Z,Y,X). Got {sph_lab.shape} at {sph_path}")
            if sph_lab.shape != masks_filtered.shape:
                raise ValueError(f"Shape mismatch: cells {masks_filtered.shape} vs spheroids {sph_lab.shape} ({sph_path})")

            # Ensure sequential labeling (optional)
            #sph_lab = sklabel(sph_lab > 0, connectivity=3)

            # --- Map each cell to its spheroid (or 0 if outside) ---
                    # --- Map each cell to its spheroid (or 0 if outside) ---
            df_map = map_cells_to_spheroids(masks_filtered, sph_lab)  # cell_label, spheroid_label, overlap_vox, frac_of_cell, frac_of_spheroid

            df_map = df_map.rename(columns={"cell_label": "ObjectID"})
            df = df.merge(df_map, on="ObjectID", how="left")

            df.to_csv(out_csv, index=False)
            print("Saved:", out_csv)

        # ----- IMAGE 1: 7-slice mosaic with rows (a)->(c)->(b)->(d) -----
        zs = seven_slices(Z)

        # Precompute normalized channels
        dapi_norm = percentile_norm(img[:, CH_DAPI, :, :])
        phal_norm = percentile_norm(img[:, CH_PHAL, :, :])
        g_norm   = percentile_norm(img[:, CH_GFP,  :, :])
        r_norm   = percentile_norm(img[:, CH_RFP,  :, :])

        fig, axs = plt.subplots(4, 7, figsize=(21, 12))
        axs = axs.reshape(4, 7)

        # Row (a): DAPI (blue) + Phalloidin (gray)
        for i, z in enumerate(zs):
            ax = axs[0, i]
            rgb = np.zeros((Y, X, 3), float)
            rgb[..., :] = phal_norm[z][:, :, None]   # gray base
            rgb[..., 2] = dapi_norm[z]               # blue channel
            ax.imshow(rgb, interpolation='none')
            ax.set_axis_off()
            ax.set_title(f"Z={z}")

        # Row (b): base DAPI (gray) + predicted masks (pre-filter)
        rng = np.random.default_rng(0)
        col_table = rng.random((max(1, masks_prefilter.max()+1), 3))
        col_table[0] = 0.0
        for i, z in enumerate(zs):
            ax = axs[1, i]
            ax.imshow(dapi_norm[z], cmap='gray', interpolation='none')
            overlay = col_table[masks_prefilter[z]]
            ax.imshow(overlay, interpolation='none', alpha=0.55)
            ax.set_axis_off()

        # Row (c): GFP (green) + RFP (red)
        for i, z in enumerate(zs):
            ax = axs[2, i]
            rgb = np.stack([r_norm[z], g_norm[z], np.zeros_like(g_norm[z])], axis=-1)
            ax.imshow(rgb, interpolation='none')
            ax.set_axis_off()

        # Row (d): base DAPI (gray) + gated masks (GFP+=green, RFP+=red, double+=yellow)
        for i, z in enumerate(zs):
            ax = axs[3, i]
            ax.imshow(dapi_norm[z], cmap='gray', interpolation='none')
            sl = labeled_mask[z]
            overlay = np.zeros((Y, X, 3), float)
            for lab_id in range(1, labeled_mask.max()+1):
                pix = (sl == lab_id)
                if not pix.any(): 
                    continue
                rpos = (lab_id-1) < len(Rpos) and Rpos[lab_id-1]
                gpos = (lab_id-1) < len(Gpos) and Gpos[lab_id-1]
                if rpos and gpos:
                    overlay[pix] = (1.0, 1.0, 0.0)      # yellow
                elif rpos:
                    overlay[pix] = (1.0, 0.0, 0.0)      # red
                elif gpos:
                    overlay[pix] = (0.0, 1.0, 0.0)      # green
            ax.imshow(overlay, interpolation='none', alpha=0.45)
            ax.set_axis_off()

        # Row labels (figure coords so they always show)
        fig.text(0.02, 0.86, "(a) DAPI blue + Phalloidin gray", rotation=90, va='center', ha='left')
        fig.text(0.02, 0.64, "(b) DAPI gray + predicted masks", rotation=90, va='center', ha='left')
        fig.text(0.02, 0.42, "(c) GFP green + RFP red",        rotation=90, va='center', ha='left')
        fig.text(0.02, 0.20, f"(d) DAPI gray + {gate_desc}",   rotation=90, va='center', ha='left')
        fig.suptitle(f"{sample} — 7-slice mosaics", y=0.98, fontsize=12)
        fig.tight_layout(rect=(0.06, 0.04, 1, 0.95))  # leave space for labels/suptitle

        png1 = os.path.join(out_dir, f"{sample}_mosaic_a-c-b-d.png")
        savefig_close(fig, png1)
        print("Saved:", png1)

        del dapi_norm, phal_norm, g_norm, r_norm

        # ----- IMAGE 2: Scatter (GFP vs RFP) with gate line -----
        if gfp_means.size:
            pth = globals().get("pos_thresholds", None)
            has_thresh = (isinstance(pth, (tuple, list)) and len(pth) == 2
                        and np.isfinite(pth[0]) and np.isfinite(pth[1]))
            fig, ax = plt.subplots(figsize=(5, 4.2))
            if has_thresh:
                xth, yth = float(pth[0]), float(pth[1])
                lim = max(1.0, np.nanmax(gfp_means), np.nanmax(rfp_means), xth, yth)
                ax.axvline(xth, linestyle='--', linewidth=1)
                ax.axhline(yth, linestyle='--', linewidth=1)
                ax.set_title(f"GFP vs RFP (thresholds x≥{xth:g}, y≥{yth:g})")
                out = os.path.join(out_dir, f"{sample}_scatter_GFP_vs_RFP_thresh_x{int(round(xth))}_y{int(round(yth))}.png")
            else:
                lim = max(1.0, np.nanmax(gfp_means), np.nanmax(rfp_means))
                ax.plot([0, lim], [0, slope_m*lim], '--', linewidth=1)
                ax.set_title(f"GFP vs RFP (slope m={slope_m})")
                out = os.path.join(out_dir, f"{sample}_scatter_GFP_vs_RFP_m{slope_m}.png")
            ax.scatter(gfp_means, rfp_means, s=6, alpha=0.35)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("Mean GFP intensity"); ax.set_ylabel("Mean RFP intensity")
            fig.tight_layout()
            savefig_close(fig, out)
            print("Saved:", out)
        else:
            print("No objects for scatter.")



        # ----- IMAGE 3: Volume histogram (pre-filter) in µm³ with minV_um3 line -----
        props_pref = skprops(sklabel(masks_prefilter > 0, connectivity=2))
        vols_vox = np.array([p.area for p in props_pref], dtype=float)

        if vols_vox.size:
            if ('voxel_size_um' in globals()) and (voxel_size_um is not None):
                z_um, y_um, x_um = map(float, voxel_size_um)
                voxel_um3 = z_um * y_um * x_um
                vols_um3 = vols_vox * voxel_um3
                xmax = max(float(np.nanmax(vols_um3)), float(minV_um3), 1.0)
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(vols_um3, bins=60, range=(0, 10000))
                ax.axvline(minV_um3, linewidth=1)
                ax.set_xlim(0, 10000); ax.set_ylim(bottom=0)
                ax.set_xlabel("Object volume (µm³)"); ax.set_ylabel("Count")
                ax.set_title("Volume distribution (pre-filter)")
                fig.tight_layout()
                png3 = os.path.join(out_dir, f"{sample}_hist_volume_prefilter_minV{int(minV_um3)}um3.png")
                savefig_close(fig, png3)
                print("Saved:", png3)
                # keep downstream filter consistent
                minV_vox = float(minV_um3) / voxel_um3
                minV = int(round(minV_vox))
                print(f"Minimum volume threshold: {minV_um3:.1f} µm³ (~{minV_vox:.1f} vox)")
            else:
                xmax = max(float(np.nanmax(vols_vox)), 1.0)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.hist(vols_vox, bins=60, range=(0, xmax))
                ax.set_xlim(0, xmax); ax.set_ylim(bottom=0)
                ax.set_xlabel("Object volume (voxels)"); ax.set_ylabel("Count")
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

        print(f"\nAll done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()


        
