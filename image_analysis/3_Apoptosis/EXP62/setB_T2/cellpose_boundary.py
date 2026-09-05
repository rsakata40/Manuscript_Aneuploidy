#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import natsort
import gc
import numpy as np
import pandas as pd
import h5py
import tifffile as tiff
import skimage.io
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table, label
from skimage.draw import line, disk as draw_disk
from skimage.color import label2rgb
from scipy.ndimage import uniform_filter, maximum_filter
from cellpose import models, core, plot

# Optional: Headless backend for batch processing (prevents windows popping up)
import matplotlib
matplotlib.use("Agg")

print('Imports complete')

# ===================================================
# 💡 GLOBAL CONFIGURATION
# ===================================================
ROOT_DIR = '/ceph.groups/mshahbazi.grp/rsakata/EXP62/LIF2TIF/setB_T2'   # Input folder
OUT_DIR = '/ceph.groups/mshahbazi.grp/rsakata/EXP62/output/setB_T2' # Output folder

os.makedirs(OUT_DIR, exist_ok=True)

# Channel Settings (0-based after reordering to ZCXY)
nucchannel = 0      # DAPI
optionalchannel = 1 # phalloidin
extractchannels = [optionalchannel, nucchannel] # Channels to pass to Cellpose

# Analysis Settings
masks_h5 = 0
slice2check = 10    # Slice index for preview plots
minsize2del = 10
cellproblimit = 0 
flowlimit = 0.4
manualD = None
threshold = 3.0     # Threshold for GFP classification
line_width = 5      # Ribbon width for profile extraction

# ===================================================
# ---------------- HELPER FUNCTIONS ----------------
def reorder_tiff_to_zcxy(img):
    """Infer axes and reorder to (Z,C,Y,X)."""
    shape = img.shape
    if img.ndim != 4:
        raise ValueError(f"Expected a 4D TIFF file, but got shape {shape}")
    dim_map = {}
    for i, dim in enumerate(shape):
        if dim < 10: dim_map['C'] = i
        elif 7 < dim < 300: dim_map['Z'] = i
        else:
            if 'Y' not in dim_map: dim_map['Y'] = i
            else: dim_map['X'] = i
    new_order = [dim_map['Z'], dim_map['C'], dim_map['Y'], dim_map['X']]
    return np.transpose(img, axes=new_order)

def sample_from_filename(fname_base):
    """Clean filename to get a sample name."""
    s = fname_base
    if s.lower().endswith('.tiff'): s = s[:-5]
    elif s.lower().endswith('.tif'): s = s[:-4]
    if ' - ' in s:
        return s.split(' - ', 1)[1].strip()
    return s.strip()

# ---------------- INITIALIZE MODEL (ONCE) ----------------
print("Initializing Cellpose model...")
use_GPU = core.use_gpu()
model = models.CellposeModel(gpu=use_GPU)
print(f"Model loaded. GPU: {use_GPU}")

# ---------------- MAIN BATCH PROCESS ----------------
def main():
    t0 = time.time()
    
    # Find all TIF/TIFF files
    files = sorted(glob.glob(os.path.join(ROOT_DIR, "*.tif")) + glob.glob(os.path.join(ROOT_DIR, "*.tiff")),
                   key=lambda p: natsort.natsort_key(p))
    print(f"Found {len(files)} files to process in {ROOT_DIR}")

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        sample = sample_from_filename(fname)
        
        print(f"\n[{i+1}/{len(files)}] Processing: {sample}")
        
        try:
            process_single_file(fpath, sample)
        except Exception as e:
            print(f"!!! Error processing {sample}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll done in {time.time()-t0:.1f}s")

def process_single_file(img_path, sample_name):
    # ---------------- PREPARE OUTPUT SUBFOLDER ----------------
    # Create a subfolder for this specific sample
    sample_out_dir = os.path.join(OUT_DIR, sample_name)
    os.makedirs(sample_out_dir, exist_ok=True)
    
    # Check if main data CSV exists to optionally skip
    out_csv = os.path.join(sample_out_dir, f"{sample_name}_data.csv")
    if os.path.exists(out_csv):
         print(f"  [Skip] Output exists in: {sample_out_dir}")
         return

    # ---------------- LOAD IMAGE ----------------
    # ext = os.path.splitext(img_path)[1].lower()
    # if ext in ('.tif', '.tiff'):
    #     img = skimage.io.imread(img_path)
    #     img = reorder_tiff_to_zcxy(img)
    # elif ext == '.h5':
    #     with h5py.File(img_path, 'r') as f:
    #         img = f['/exported_data'][:]
    #     img = reorder_tiff_to_zcxy(img)
    # else:
    #     print(f"Unsupported format: {ext}")
    #     return

    # Z, C, Y, X = img.shape
    img_raw = skimage.io.imread(img_path)
    img = img_raw
    Z, C, Y, X = img.shape
    
    # ----------------- OPTIMIZED BATCH SEGMENTATION -----------------
    print(f"  > Segmenting {Z} slices...")

    # 1. Create list of 2D slices (C, Y, X) for segmentation channels
    imgs_list = [img[z, extractchannels, :, :] for z in range(Z)]

    # 2. Run model
    masks_list, flows_list, styles_list = model.eval(
        imgs_list, 
        channel_axis=0, 
        do_3D=False,
        diameter=manualD, 
        flow_threshold=flowlimit, 
        cellprob_threshold=cellproblimit,
        min_size=minsize2del, 
        progress=False
    )

    # 3. Convert back to 3D array
    masks_3d = np.array(masks_list).astype(np.int32)
    num_objects = int(masks_3d.max())
    print(f"  > Detected {num_objects} objects.")

    # ----------------- PLOT 1: Segmentation Preview -----------------
    z_idx = int(np.clip(slice2check, 0, Z-1))
    flow_for_plot = flows_list[z_idx][0] 

    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, imgs_list[z_idx], masks_3d[z_idx], flow_for_plot, channels=[0,1])
    plt.tight_layout()
    # SAVE TO SUBFOLDER
    plt.savefig(os.path.join(sample_out_dir, f"{sample_name}_segmentation_preview_z{z_idx}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---------------- ANALYSIS: FIND NEIGHBORS ----------------
    masks = masks_3d
    all_pairs = []
    cell_dapi_means = {} 

    print("  > Finding neighbors...")
    for z in range(Z):
        mask_z = masks[z]
        if mask_z.max() == 0: continue

        dapi_slice = img[z, nucchannel, :, :]

        # Regionprops
        props = regionprops_table(mask_z, intensity_image=dapi_slice, properties=("label", "centroid", "intensity_mean"))
        df = pd.DataFrame(props)
        
        centroids = {int(lab): (float(cy), float(cx)) for lab, cy, cx in zip(df["label"], df["centroid-0"], df["centroid-1"])}
        for lab, mean_int in zip(df["label"], df["intensity_mean"]):
            cell_dapi_means[(z, int(lab))] = mean_int

        # Neighbor Finding
        neighbors = set()
        expanded_mask = maximum_filter(mask_z, size=5)
        
        diff_h = (expanded_mask[:, :-1] != expanded_mask[:, 1:]) & (expanded_mask[:, :-1] != 0) & (expanded_mask[:, 1:] != 0)
        labels_h = np.stack([expanded_mask[:, :-1][diff_h], expanded_mask[:, 1:][diff_h]], axis=1)
        
        diff_v = (expanded_mask[:-1, :] != expanded_mask[1:, :]) & (expanded_mask[:-1, :] != 0) & (expanded_mask[1:, :] != 0)
        labels_v = np.stack([expanded_mask[:-1, :][diff_v], expanded_mask[1:, :][diff_v]], axis=1)
        
        all_adj = []
        if labels_h.size > 0: all_adj.append(labels_h)
        if labels_v.size > 0: all_adj.append(labels_v)
        
        if len(all_adj) > 0:
            all_adj = np.vstack(all_adj)
            unique_pairs = np.unique(np.sort(all_adj, axis=1), axis=0)
            for p in unique_pairs:
                if p[0] in centroids and p[1] in centroids:
                    neighbors.add(tuple(p))

        for lab_i, lab_j in neighbors:
            y1, x1 = centroids[lab_i]
            y2, x2 = centroids[lab_j]
            all_pairs.append({"z": z, "label_i": lab_i, "label_j": lab_j, "y1": y1, "x1": x1, "y2": y2, "x2": x2})

    # ---------------- PLOT 2: Detailed Ribbons ----------------
    z_idx_vis = z_idx # Use same Z as preview
    
    # Setup full image data
    mask_z = masks[z_idx_vis]
    img_z = img[z_idx_vis, nucchannel, :, :] 
    img_z_norm = (img_z - img_z.min()) / (img_z.max() - img_z.min())
    mask_rgb = label2rgb(mask_z, bg_label=0, bg_color=(0, 0, 0), image=img_z_norm)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(mask_rgb, interpolation="nearest")
    plt.title(f"Detailed Profile Ribbons (z={z_idx_vis})")

    pairs_in_vis_plane = [p for p in all_pairs if p["z"] == z_idx_vis]
    for pair in pairs_in_vis_plane:
        y1, x1, y2, x2 = pair["y1"], pair["x1"], pair["y2"], pair["x2"]
        rr, cc = line(int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2)))
        
        # Draw ribbon for visualization
        all_ribbon_rr, all_ribbon_cc = [], []
        for r, c in zip(rr, cc):
            rr_w, cc_w = draw_disk((r, c), radius=line_width, shape=(Y, X))
            all_ribbon_rr.extend(rr_w)
            all_ribbon_cc.extend(cc_w)
        ribbon_coords = np.unique(np.stack([all_ribbon_rr, all_ribbon_cc], axis=1), axis=0)
        
        plt.plot(ribbon_coords[:, 1], ribbon_coords[:, 0], marker='s', markersize=2, alpha=0.08, linestyle='None', color='yellow')
        plt.plot([x1, x2], [y1, y2], linewidth=1, color='red', linestyle='--')
        plt.plot(x1, y1, 'o', color='white', markersize=4)
        plt.plot(x2, y2, 'o', color='white', markersize=4)

    # Removed: plt.xlim and plt.ylim logic to show full image
    plt.axis("off")
    plt.tight_layout()
    # SAVE TO SUBFOLDER
    plt.savefig(os.path.join(sample_out_dir, f"{sample_name}_detailed_ribbons_z{z_idx_vis}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---------------- OPTIMIZED PROFILE EXTRACTION ----------------
    print("  > Extracting profiles...")
    channel_names =  ["DAPI", "phalloidin", "GFP", "ECAD", "caspase3"] # Ensure these match your image channels
    data_list = []

    for z in range(Z):
        pairs_in_z = [p for p in all_pairs if p["z"] == z]
        if not pairs_in_z: continue
        
        current_slice = img[z] # (C, Y, X)
        blur_size = (0, line_width*2+1, line_width*2+1) 
        blurred_slice = uniform_filter(current_slice.astype(float), size=blur_size)
        
        for pair in pairs_in_z:
            label_i, label_j = pair["label_i"], pair["label_j"]
            y1, x1, y2, x2 = pair["y1"], pair["x1"], pair["y2"], pair["x2"]
            
            rr, cc = line(int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2)))
            valid = (rr >= 0) & (rr < Y) & (cc >= 0) & (cc < X)
            rr, cc = rr[valid], cc[valid]
            if len(rr) == 0: continue

            intensities = blurred_slice[:, rr, cc].T 
            dist = np.arange(len(rr))
            
            di = cell_dapi_means.get((z, label_i), np.nan)
            dj = cell_dapi_means.get((z, label_j), np.nan)
            norm = np.nanmean([di, dj])
            if norm == 0 or np.isnan(norm): norm = 1.0
                
            chan_df = pd.DataFrame(intensities, columns=channel_names[:C]) # Slice names if fewer channels in IMG
            chan_df["z"] = z
            chan_df["label_i"] = label_i
            chan_df["label_j"] = label_j
            chan_df["position"] = dist
            
            chan_df_melted = chan_df.melt(id_vars=["z", "label_i", "label_j", "position"], var_name="channel", value_name="intensity")
            chan_df_melted["normalized_intensity"] = chan_df_melted["intensity"] / norm
            data_list.append(chan_df_melted)

    if not data_list:
        print(f"  > No pairs found for {sample_name}")
        return

    df_long = pd.concat(data_list, ignore_index=True)

    # ---------------- PEAK ALIGNMENT ----------------
    ph_peaks = df_long[df_long["channel"] == "phalloidin"].groupby(["z", "label_i", "label_j"])["normalized_intensity"].idxmax()
    peak_positions = df_long.loc[ph_peaks, ["z", "label_i", "label_j", "position"]].set_index(["z", "label_i", "label_j"])
    peak_positions.rename(columns={"position": "peak_pos"}, inplace=True)

    df_long = df_long.join(peak_positions, on=["z", "label_i", "label_j"])
    df_long["distance_from_peak"] = df_long["position"] - df_long["peak_pos"]

    df_long = df_long[(df_long["distance_from_peak"] >= -15) & (df_long["distance_from_peak"] <= 15)].copy()

    # ---------------- PLOTTING & CLASSIFICATION ----------------
    print("  > Generating summary plots...")
    
    # Avg Profile
    summary = df_long.groupby(["channel", "distance_from_peak"])["normalized_intensity"].agg(["mean", "std", "count"]).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    fig = plt.figure(figsize=(7, 5))
    for ch, grp in summary.groupby("channel"):
        plt.plot(grp["distance_from_peak"], grp["mean"], label=ch)
        plt.fill_between(grp["distance_from_peak"], grp["mean"]-grp["sem"], grp["mean"]+grp["sem"], alpha=0.3)
    plt.title("Average Normalized Intensity")
    plt.legend()
    # SAVE TO SUBFOLDER
    plt.savefig(os.path.join(sample_out_dir, f"{sample_name}_avg_profile.png"), dpi=150)
    plt.close(fig)

    # Threshold Histogram & Classification
    def calc_before_after(df):
        before = df[df["distance_from_peak"] < 0]["normalized_intensity"].mean()
        after = df[df["distance_from_peak"] > 0]["normalized_intensity"].mean()
        return pd.Series({"intensity_before": before, "intensity_after": after})

    gfp_df = df_long[df_long["channel"] == "GFP"]
    if not gfp_df.empty:
        pair_stats = gfp_df.groupby(["z", "label_i", "label_j"]).apply(calc_before_after).reset_index()

        fig = plt.figure(figsize=(10, 5))
        plt.hist(pair_stats["intensity_before"].dropna(), bins=50, alpha=0.5, label="Before")
        plt.hist(pair_stats["intensity_after"].dropna(), bins=50, alpha=0.5, label="After")
        plt.legend()
        plt.title("GFP Intensity Distribution")
        # SAVE TO SUBFOLDER
        plt.savefig(os.path.join(sample_out_dir, f"{sample_name}_threshold_histogram.png"), dpi=150)
        plt.close(fig)

        # Classification
        pair_stats["cell_cell"] = "rev_ctr"
        cond_ctr = (pair_stats["intensity_before"] < threshold) & (pair_stats["intensity_after"] < threshold)
        cond_rev = (pair_stats["intensity_before"] > threshold) & (pair_stats["intensity_after"] > threshold)
        pair_stats.loc[cond_ctr, "cell_cell"] = "ctr_ctr"
        pair_stats.loc[cond_rev, "cell_cell"] = "rev_rev"

        # Merge classification back
        df_final = df_long.merge(pair_stats[["z", "label_i", "label_j", "cell_cell"]], on=["z", "label_i", "label_j"], how="left")
    else:
        df_final = df_long
        df_final["cell_cell"] = "unknown"

    # Save Data TO SUBFOLDER
    df_final.to_csv(os.path.join(sample_out_dir, f"{sample_name}_data.csv"), index=False)

    # Grouped Profiles
    summary_grouped = df_final.groupby(["cell_cell", "channel", "distance_from_peak"])["normalized_intensity"].agg(["mean", "std", "count"]).reset_index()
    summary_grouped["sem"] = summary_grouped["std"] / np.sqrt(summary_grouped["count"])
    cell_classes = sorted(summary_grouped["cell_cell"].unique())

    if len(cell_classes) > 0:
        fig, axes = plt.subplots(1, len(cell_classes), figsize=(6 * len(cell_classes), 5), sharey=True)
        if len(cell_classes) == 1: axes = [axes]

        for ax, cls in zip(axes, cell_classes):
            sub = summary_grouped[summary_grouped["cell_cell"] == cls]
            for ch, grp in sub.groupby("channel"):
                if ch in ["phalloidin", "GFP", "ECAD", "DAPI"]:
                    ax.plot(grp["distance_from_peak"], grp["mean"], label=ch)
                    ax.fill_between(grp["distance_from_peak"], grp["mean"]-grp["sem"], grp["mean"]+grp["sem"], alpha=0.3)
            ax.set_title(cls)
            ax.legend()
        plt.tight_layout()
        # SAVE TO SUBFOLDER
        plt.savefig(os.path.join(sample_out_dir, f"{sample_name}_grouped_profiles.png"), dpi=150)
        plt.close(fig)

    # Combined Profile per Channel
    channels_to_combine = ["ECAD", "phalloidin"]
    for target_channel in channels_to_combine:
        summary_ch = summary_grouped[summary_grouped["channel"] == target_channel].copy()
        if summary_ch.empty: continue

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        for cell_state in cell_classes:
            sub_state = summary_ch[summary_ch["cell_cell"] == cell_state].sort_values("distance_from_peak")
            ax.plot(sub_state["distance_from_peak"], sub_state["mean"], label=cell_state, linewidth=2)
            ax.fill_between(sub_state["distance_from_peak"], sub_state["mean"] - sub_state["sem"], sub_state["mean"] + sub_state["sem"], alpha=0.2)

        ax.axvline(0, linestyle="--", color='gray', alpha=0.7)
        ax.set_title(f"Normalized {target_channel} Profile")
        ax.legend(title="Cell Pair Class")
        plt.tight_layout()
        # SAVE TO SUBFOLDER
        plt.savefig(os.path.join(sample_out_dir, f"{sample_name}_combined_profile_{target_channel}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Force cleanup to prevent memory leaks in batch
    gc.collect()

if __name__ == "__main__":
    main()