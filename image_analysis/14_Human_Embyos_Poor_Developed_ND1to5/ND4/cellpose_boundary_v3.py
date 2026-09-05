#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cellpose-based 2D cell-interface profiling across a Z stack.

Changes in this revision
------------------------
1. Voxel size is an explicit constant (same format as cellpose_batch.py), with a
   non-binding cross-check against TIFF metadata that only warns.
2. Axis reordering uses shape ranking rather than size thresholds, so cropped
   fields, thin stacks and long stacks all work.
3. Distance zero is the geometric interface from the Cellpose masks. No
   intensity-based peak alignment, so no selection-on-noise bias. The offset
   between the geometric zero and the QC channel's local maximum is measured
   and reported, but never applied.
4. Profile half-length is clipped per pair to a fraction of the centroid-centroid
   distance, so profiles cannot run through a third cell when cells are small.
5. Ribbon width validity and interface contact length are used as filters, not
   just recorded.
6. Distances are keyed on an integer step index, so any PROFILE_STEP_UM works.
7. No thresholding or classification. Per-pair side intensities are written for
   every channel so classification can be done downstream.
8. Hot paths vectorised: one float conversion per slice, one map_coordinates call
   per channel per chunk of interfaces, vectorised interface detection, chunked
   Cellpose evaluation.
"""

import matplotlib
matplotlib.use("Agg")

import gc
import glob
import os
import time
import traceback

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import natsort
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import distance_transform_edt, map_coordinates
from skimage.color import label2rgb
from skimage.measure import regionprops_table
from cellpose import core, models

plt.ioff()


# ============================================================================
# USER CONFIGURATION
# ============================================================================
ROOT_DIR = "/ceph.groups/mshahbazi.grp/rsakata/HE_Nondeveloping4/LIF2TIF"
OUT_DIR = "/ceph.groups/mshahbazi.grp/rsakata/HE_Nondeveloping4/output/cellpose_boundary3"

# Channel names in the TIFF after reordering to Z,C,Y,X.
CHANNEL_ORDER = ["DAPI", "GATA4", "NANOG", "GATA3", "ECAD"]
NUCLEAR_CHANNEL = "DAPI"
SEGMENTATION_CHANNELS = ["ECAD", "DAPI"]

# (z, y, x) in micrometres. Same format as cellpose_batch.py. Only y and x are
# used here; z is kept so the tuple can be shared between scripts.
VOXEL_SIZE_UM = (1.0, 0.5687380, 0.5687380)
WARN_ON_VOXEL_MISMATCH = True

# Explicit axes string used only if the file's own metadata is unusable.
INPUT_AXES = None               # e.g. "ZCYX" or "CZYX"

# Cellpose settings for the cell-boundary masks (2D, independently per Z slice).
CELLPROB_THRESHOLD = 0.0
FLOW_THRESHOLD = 0.4
DIAMETER_PX = None
MIN_OBJECT_AREA_UM2 = 10.0
EVAL_CHUNK_SLICES = 16          # z-slices per model.eval call

# Separate DAPI-mask segmentation used only for normalization. Each DAPI object
# is assigned to the cell mask with which it has the largest pixel overlap.
# Interface pairs are discarded unless BOTH cells have an assigned DAPI mask.
DAPI_CELLPROB_THRESHOLD = 0.0
DAPI_FLOW_THRESHOLD = 0.4
DAPI_DIAMETER_PX = None
MIN_DAPI_MASK_AREA_UM2 = 10.0
MIN_DAPI_CELL_OVERLAP_FRACTION = 0.50

# Interface detection.
NEIGHBOR_EXPANSION_UM = 2.0     # each label grows by this; gaps up to 2x are bridged
MIN_INTERFACE_LENGTH_UM = 1.0   # discard pairs whose expanded interface is shorter
# Maximum edge-to-edge gap between the ORIGINAL masks for a pair to count as a
# real contact. Pairs bridged only by label expansion across a wider gap are
# dropped: their "interface" is background, not a membrane.
MAX_PAIR_GAP_UM = 1.0

# Profile geometry.
PROFILE_HALF_LENGTH_UM = 5    # absolute cap on each side of the interface
# Each side is clipped separately, relative to the distance from the interface
# to that side's centroid. The interface is usually not halfway between the two
# centroids, so a symmetric clip overshoots into the smaller cell.
PROFILE_LENGTH_FRACTION = 1.6   # x distance from interface to that side's centroid
PROFILE_HALF_WIDTH_UM = 1.5
PROFILE_STEP_UM = 0.5
MIN_VALID_WIDTH_FRACTION = 0.8  # drop positions whose ribbon is mostly off-image
# Ribbon samples landing in a cell other than the pair are excluded from the
# width-average; positions exceeding this fraction are dropped entirely.
# Background is allowed and recorded, since a real membrane can sit just
# outside both Cellpose masks.
MAX_THIRD_CELL_FRACTION = 0.05
RIBBON_CHUNK = 512              # interfaces sampled per map_coordinates call

# Normalization. "pair_nuclear_dapi_mean" divides each ribbon intensity by the
# mean of the two connected cells' DAPI-mask mean intensities. Pairs lacking a
# DAPI mask for either cell are removed before sampling. "none" keeps raw values.
NORMALIZATION = "pair_nuclear_dapi_mean"

# Quality control only. The offset between the geometric zero and this channel's
# local maximum is reported, never subtracted. Set to None to skip.
ALIGNMENT_QC_CHANNEL = "ECAD"
ALIGNMENT_QC_SEARCH_UM = 4.0

# Output.
PREVIEW_Z = None                # None = middle slice
SKIP_EXISTING = True
WRITE_PROFILE_TABLE = True      # the large long-format table
WRITE_SUMMARY_TABLE = True      # mean/sem per channel per distance

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# GENERAL HELPERS
# ============================================================================
def savefig_close(fig, outpath, dpi=200):
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def sample_from_filename(fname_base):
    s = fname_base
    lower = s.lower()
    if lower.endswith(".tiff"):
        s = s[:-5]
    elif lower.endswith(".tif"):
        s = s[:-4]
    if " - " in s:
        return s.split(" - ", 1)[1].strip()
    return s.strip()


def percentile_norm(img2d, p_low=1.0, p_high=99.0):
    arr = np.asarray(img2d, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, (p_low, p_high))
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def image_figsize(y_size, x_size, ncols=1, panel_height=5.0):
    aspect = x_size / max(y_size, 1)
    panel_width = float(np.clip(panel_height * aspect, 3.5, 9.0))
    return panel_width * ncols, panel_height


def _apply_axes(arr, axes):
    """Reorder using an explicit axes string. Raises if it cannot be honoured."""
    axes = list(axes.upper())
    if len(axes) != arr.ndim:
        raise ValueError(f"axes {''.join(axes)!r} does not match ndim {arr.ndim}")
    if "C" not in axes and "S" in axes:
        axes[axes.index("S")] = "C"
    for i in range(len(axes) - 1, -1, -1):
        if axes[i] not in {"Z", "C", "Y", "X"}:
            if arr.shape[i] != 1:
                raise ValueError(
                    f"non-singleton axis {axes[i]!r} of size {arr.shape[i]}"
                )
            arr = np.squeeze(arr, axis=i)
            axes.pop(i)
    if len(set(axes)) != len(axes):
        raise ValueError(f"repeated axes {''.join(axes)!r}")
    if not {"Y", "X"}.issubset(axes):
        raise ValueError(f"no Y/X in {''.join(axes)!r}")
    order = [axes.index(a) for a in "ZCYX" if a in axes]
    arr = np.transpose(arr, axes=order)
    if "C" not in axes:
        arr = arr[:, None] if "Z" in axes else arr[None, None]
    elif "Z" not in axes:
        arr = arr[None]
    return arr


def _infer_axes_by_shape(arr, n_channels):
    """
    Last-resort inference. Y and X are the two largest axes and C matches the
    channel count. Refuses when Z could be as large as a spatial axis, because
    that case is genuinely ambiguous and would otherwise transpose silently.
    """
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D stack for shape inference, got {arr.shape}")
    shape = arr.shape
    spatial = sorted(sorted(range(4), key=lambda i: shape[i], reverse=True)[:2])
    rest = [i for i in range(4) if i not in spatial]
    exact = [i for i in rest if shape[i] == n_channels]
    if len(exact) != 1:
        raise ValueError(
            f"Cannot identify the channel axis in shape {shape}: expected exactly one "
            f"axis of size {n_channels}, found sizes {[shape[i] for i in rest]}."
        )
    c_axis = exact[0]
    z_axis = [i for i in rest if i != c_axis][0]
    y_axis, x_axis = spatial
    if shape[z_axis] >= min(shape[y_axis], shape[x_axis]):
        raise ValueError(
            f"Ambiguous shape {shape}: the inferred Z axis (size {shape[z_axis]}) is not "
            f"smaller than the inferred Y/X axes ({shape[y_axis]}, {shape[x_axis]}). "
            f"Set INPUT_AXES explicitly, for example INPUT_AXES='ZCYX'."
        )
    return np.transpose(arr, axes=[z_axis, c_axis, y_axis, x_axis])


def read_tiff_zcxy(path):
    """
    Load a TIFF as (Z,C,Y,X).

    Order of preference: the file's own axes metadata, then INPUT_AXES, then
    shape inference. Shape inference refuses ambiguous cases rather than
    guessing, because a silent transpose corrupts every downstream number.
    """
    with tiff.TiffFile(path) as tf:
        arr = tf.asarray()
        meta_axes = (getattr(tf.series[0], "axes", "") or "").upper()

    attempts = [("file axes metadata", meta_axes)]
    if INPUT_AXES:
        attempts.append(("INPUT_AXES", INPUT_AXES.upper()))
    for source, axes in attempts:
        if not axes:
            continue
        try:
            out = _apply_axes(arr, axes)
            print(f"  > Axes from {source} ({axes}): {arr.shape} -> {out.shape} (Z,C,Y,X)")
            return out
        except ValueError as exc:
            print(f"  [Warning] {source} ({axes!r}) unusable: {exc}")

    out = _infer_axes_by_shape(arr, len(CHANNEL_ORDER))
    print(f"  [Warning] Falling back to shape inference: {arr.shape} -> {out.shape} (Z,C,Y,X)")
    return out


def _rational_to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        numerator, denominator = value
        return float(numerator) / float(denominator)


def check_voxel_size_against_file(path, voxel_um, tol=0.05):
    """Warn if the file's own metadata disagrees with VOXEL_SIZE_UM. Never overrides."""
    if not WARN_ON_VOXEL_MISMATCH:
        return
    y_cfg, x_cfg = float(voxel_um[1]), float(voxel_um[2])
    found = None
    try:
        with tiff.TiffFile(path) as tf:
            ij = tf.imagej_metadata or {}
            unit = str(ij.get("unit", "")).strip().lower().replace("\u03bc", "\u00b5")
            factor = {"micron": 1.0, "microns": 1.0, "um": 1.0, "\u00b5m": 1.0,
                      "nm": 1e-3, "mm": 1e3}.get(unit)
            x_tag = tf.pages[0].tags.get("XResolution")
            y_tag = tf.pages[0].tags.get("YResolution")
            if factor and x_tag is not None and y_tag is not None:
                x_res = _rational_to_float(x_tag.value)
                y_res = _rational_to_float(y_tag.value)
                if x_res > 0 and y_res > 0:
                    found = (factor / y_res, factor / x_res)
            if found is None and getattr(tf, "ome_metadata", None):
                import re as _re
                ome = tf.ome_metadata
                units = {"\u00b5m": 1.0, "um": 1.0, "micron": 1.0, "nm": 1e-3, "mm": 1e3}
                vals = {}
                for key in ("X", "Y"):
                    mv = _re.search(rf'PhysicalSize{key}="([0-9.eE+-]+)"', ome)
                    mu = _re.search(rf'PhysicalSize{key}Unit="([^"]+)"', ome)
                    if mv:
                        u = (mu.group(1) if mu else "\u00b5m").strip().lower().replace("\u03bc", "\u00b5")
                        vals[key] = float(mv.group(1)) * units.get(u, 1.0)
                if "X" in vals and "Y" in vals and vals["X"] > 0 and vals["Y"] > 0:
                    found = (vals["Y"], vals["X"])
    except Exception:
        return
    if found is None:
        return
    if abs(found[0] - y_cfg) / y_cfg > tol or abs(found[1] - x_cfg) / x_cfg > tol:
        print(
            f"  [WARNING] file metadata says {found[0]:.4f} x {found[1]:.4f} µm/px but "
            f"VOXEL_SIZE_UM says {y_cfg:.4f} x {x_cfg:.4f}. Every µm setting is scaled "
            f"by {y_cfg / found[0]:.2f}x. Update VOXEL_SIZE_UM if this is a different objective."
        )
    else:
        print(f"  > Voxel size {y_cfg:.4f} µm/px agrees with file metadata.")


def channel_map_for_image(img):
    c_count = img.shape[1]
    if c_count != len(CHANNEL_ORDER):
        raise ValueError(
            f"Image has {c_count} channels but CHANNEL_ORDER has {len(CHANNEL_ORDER)}: {CHANNEL_ORDER}"
        )
    if len(set(CHANNEL_ORDER)) != len(CHANNEL_ORDER):
        raise ValueError(f"CHANNEL_ORDER contains duplicates: {CHANNEL_ORDER}")
    return {name: i for i, name in enumerate(CHANNEL_ORDER)}


def choose_preview_z(z_count, interface_counts=None):
    """Prefer the slice with the most retained interfaces, so the QC image is informative."""
    if PREVIEW_Z is not None:
        return int(np.clip(PREVIEW_Z, 0, z_count - 1))
    if interface_counts:
        best = max(interface_counts, key=lambda z: interface_counts[z])
        if interface_counts[best] > 0:
            return best
    return z_count // 2


# ============================================================================
# INTERFACE DETECTION (vectorised)
# ============================================================================
def physically_expand_labels(mask, expansion_um, pixel_size_um_yx):
    """
    Grow labels into background by nearest labelled pixel. Never overwrites labels.
    Also returns the physical distance of every pixel to its nearest labelled pixel,
    which is used to recover the true edge-to-edge gap between mask pairs.
    """
    mask = np.asarray(mask, dtype=np.int32)
    distances_um = np.zeros(mask.shape, dtype=np.float64)
    if mask.max() == 0 or expansion_um <= 0:
        return mask.copy(), distances_um
    background = mask == 0
    distances_um, nearest = distance_transform_edt(
        background, sampling=pixel_size_um_yx, return_indices=True
    )
    nearest_labels = mask[tuple(nearest)]
    expanded = mask.copy()
    fill = background & (distances_um <= expansion_um) & (nearest_labels > 0)
    expanded[fill] = nearest_labels[fill]
    return expanded, distances_um


def find_interfaces(mask_z, pixel_size_um_yx):
    """
    Return a DataFrame of neighbouring label pairs.

    Vectorised: no Python loop over boundary pixels. For each pair the contact
    point closest to the centroid midpoint is selected via a lexsort.

    Note on naming: because labels are expanded before adjacency is tested,
    `expanded_interface_length_um` is the length of the border between the
    expanded nearest-label regions, NOT a verified membrane contact. `gap_um`
    is the true minimum edge-to-edge distance between the original masks, and
    pairs with gap_um > MAX_PAIR_GAP_UM are discarded.
    """
    cols = ["label_i", "label_j", "y1", "x1", "y2", "x2", "interface_y", "interface_x",
            "n_interface_px", "expanded_interface_length_um", "gap_um",
            "centroid_distance_um", "dist_to_i_um", "dist_to_j_um"]
    empty = pd.DataFrame(columns=cols)
    if mask_z.max() == 0:
        return empty, 0

    props = pd.DataFrame(
        regionprops_table(mask_z, properties=("label", "centroid"))
    )
    if props.empty:
        return empty, 0

    y_um, x_um = map(float, pixel_size_um_yx)
    max_label = int(mask_z.max())
    cy = np.full(max_label + 1, np.nan)
    cx = np.full(max_label + 1, np.nan)
    labels = props["label"].to_numpy(int)
    cy[labels] = props["centroid-0"].to_numpy(float)
    cx[labels] = props["centroid-1"].to_numpy(float)

    expanded, dist_to_mask_um = physically_expand_labels(
        mask_z, NEIGHBOR_EXPANSION_UM, pixel_size_um_yx)

    ys_all, xs_all, a_all, b_all, len_all, gap_all = [], [], [], [], [], []
    # left-right transitions: the shared edge is vertical, so its length is y_um
    a, b = expanded[:, :-1], expanded[:, 1:]
    v = (a != b) & (a > 0) & (b > 0)
    if v.any():
        yy, xx = np.nonzero(v)
        ys_all.append(yy.astype(np.float64))
        xs_all.append(xx.astype(np.float64) + 0.5)
        a_all.append(a[v]); b_all.append(b[v])
        len_all.append(np.full(yy.size, y_um))
        gap_all.append(dist_to_mask_um[:, :-1][v] + dist_to_mask_um[:, 1:][v])
    # up-down transitions: the shared edge is horizontal, so its length is x_um
    a, b = expanded[:-1, :], expanded[1:, :]
    v = (a != b) & (a > 0) & (b > 0)
    if v.any():
        yy, xx = np.nonzero(v)
        ys_all.append(yy.astype(np.float64) + 0.5)
        xs_all.append(xx.astype(np.float64))
        a_all.append(a[v]); b_all.append(b[v])
        len_all.append(np.full(yy.size, x_um))
        gap_all.append(dist_to_mask_um[:-1, :][v] + dist_to_mask_um[1:, :][v])
    if not ys_all:
        return empty, 0

    ys = np.concatenate(ys_all); xs = np.concatenate(xs_all)
    edge_len = np.concatenate(len_all); edge_gap = np.concatenate(gap_all)
    la = np.concatenate(a_all).astype(np.int64)
    lb = np.concatenate(b_all).astype(np.int64)
    lo = np.minimum(la, lb); hi = np.maximum(la, lb)
    code = lo * (max_label + 1) + hi

    mid_y = 0.5 * (cy[lo] + cy[hi])
    mid_x = 0.5 * (cx[lo] + cx[hi])
    d2 = (ys - mid_y) ** 2 + (xs - mid_x) ** 2
    d2 = np.where(np.isfinite(d2), d2, np.inf)

    order = np.lexsort((d2, code))
    code_s = code[order]
    starts = np.flatnonzero(np.r_[True, code_s[1:] != code_s[:-1]])
    ends = np.r_[starts[1:], code_s.size]
    counts = ends - starts
    best = order[starts]

    # per-pair aggregates over the group's contact pixels
    len_s = edge_len[order]; gap_s = edge_gap[order]
    total_len = np.add.reduceat(len_s, starts)
    min_gap = np.minimum.reduceat(gap_s, starts)

    pair_lo = lo[best]; pair_hi = hi[best]
    iy = ys[best]; ix = xs[best]
    df = pd.DataFrame({
        "label_i": pair_lo,
        "label_j": pair_hi,
        "y1": cy[pair_lo], "x1": cx[pair_lo],
        "y2": cy[pair_hi], "x2": cx[pair_hi],
        "interface_y": iy, "interface_x": ix,
        "n_interface_px": counts,
        "expanded_interface_length_um": total_len,
        "gap_um": min_gap,
    })
    df["centroid_distance_um"] = np.hypot(
        (df["y2"] - df["y1"]) * y_um, (df["x2"] - df["x1"]) * x_um)
    # distance from the interface to each centroid; the interface is generally
    # NOT halfway between them, so the two sides are clipped independently
    df["dist_to_i_um"] = np.hypot((iy - df["y1"]) * y_um, (ix - df["x1"]) * x_um)
    df["dist_to_j_um"] = np.hypot((df["y2"] - iy) * y_um, (df["x2"] - ix) * x_um)

    finite = np.isfinite(df["y1"]) & np.isfinite(df["y2"])
    long_enough = df["expanded_interface_length_um"] >= MIN_INTERFACE_LENGTH_UM
    close_enough = df["gap_um"] <= MAX_PAIR_GAP_UM
    nondegenerate = df["centroid_distance_um"] > 1e-6
    keep = finite & long_enough & close_enough & nondegenerate
    n_gap = int((finite & long_enough & nondegenerate & ~close_enough).sum())
    n_short = int((finite & nondegenerate & ~long_enough).sum())
    return df[keep].reset_index(drop=True), int(len(df))


def map_dapi_masks_to_cells(cell_mask, dapi_mask, dapi_image):
    """
    Assign each DAPI mask to one cell mask by maximum pixel overlap.

    A DAPI object is accepted only when at least
    MIN_DAPI_CELL_OVERLAP_FRACTION of its pixels overlap the assigned cell.
    If multiple DAPI masks map to one cell, the assignment with the largest
    overlap fraction (then largest overlap pixel count) is retained.

    Returns
    -------
    assignments : DataFrame
        One row per cell with an accepted DAPI mask.
    stats : dict
        Counts used for per-slice QC reporting.
    """
    columns = [
        "cell_label", "dapi_mask_label", "nuclear_dapi_mean",
        "nucleus_area_px", "overlap_pixels", "overlap_fraction",
    ]
    empty = pd.DataFrame(columns=columns)
    if cell_mask.max() == 0 or dapi_mask.max() == 0:
        return empty, {
            "n_dapi_masks": int(dapi_mask.max()),
            "n_overlap_accepted": 0,
            "n_cells_assigned": 0,
            "n_duplicate_cell_assignments": 0,
        }

    props = pd.DataFrame(
        regionprops_table(
            dapi_mask,
            intensity_image=dapi_image,
            properties=("label", "area", "intensity_mean"),
        )
    )
    if props.empty:
        return empty, {
            "n_dapi_masks": 0,
            "n_overlap_accepted": 0,
            "n_cells_assigned": 0,
            "n_duplicate_cell_assignments": 0,
        }

    rows = []
    for row in props.itertuples(index=False):
        nucleus_label = int(row.label)
        nucleus_area = int(row.area)
        nucleus_pixels = dapi_mask == nucleus_label
        overlapping_cells = cell_mask[nucleus_pixels]
        overlapping_cells = overlapping_cells[overlapping_cells > 0]
        if overlapping_cells.size == 0 or nucleus_area <= 0:
            continue

        cell_labels, counts = np.unique(overlapping_cells, return_counts=True)
        best_idx = int(np.argmax(counts))
        cell_label = int(cell_labels[best_idx])
        overlap_pixels = int(counts[best_idx])
        overlap_fraction = overlap_pixels / float(nucleus_area)
        if overlap_fraction < MIN_DAPI_CELL_OVERLAP_FRACTION:
            continue

        rows.append({
            "cell_label": cell_label,
            "dapi_mask_label": nucleus_label,
            "nuclear_dapi_mean": float(row.intensity_mean),
            "nucleus_area_px": nucleus_area,
            "overlap_pixels": overlap_pixels,
            "overlap_fraction": overlap_fraction,
        })

    if not rows:
        return empty, {
            "n_dapi_masks": len(props),
            "n_overlap_accepted": 0,
            "n_cells_assigned": 0,
            "n_duplicate_cell_assignments": 0,
        }

    assignments = pd.DataFrame(rows)
    n_overlap_accepted = len(assignments)
    n_duplicate = int(assignments.duplicated("cell_label", keep=False).sum())
    assignments = (
        assignments.sort_values(
            ["cell_label", "overlap_fraction", "overlap_pixels", "nucleus_area_px"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates("cell_label", keep="first")
        .reset_index(drop=True)
    )
    return assignments, {
        "n_dapi_masks": len(props),
        "n_overlap_accepted": n_overlap_accepted,
        "n_cells_assigned": len(assignments),
        "n_duplicate_cell_assignments": n_duplicate,
    }


def attach_nuclear_dapi_to_interfaces(interfaces, assignments):
    """
    Add the two cells' DAPI-mask measurements and remove incomplete pairs.

    No fallback normalization is used: a pair is retained only when both cell
    labels have an accepted DAPI mask with a finite positive mean DAPI value.
    """
    if interfaces.empty or assignments.empty:
        return interfaces.iloc[0:0].copy(), len(interfaces)

    by_cell = assignments.set_index("cell_label")
    out = interfaces.copy()
    for suffix, label_col in (("i", "label_i"), ("j", "label_j")):
        out[f"dapi_mask_{suffix}"] = out[label_col].map(by_cell["dapi_mask_label"])
        out[f"nuclear_dapi_mean_{suffix}"] = out[label_col].map(by_cell["nuclear_dapi_mean"])
        out[f"dapi_overlap_fraction_{suffix}"] = out[label_col].map(by_cell["overlap_fraction"])
        out[f"dapi_nucleus_area_px_{suffix}"] = out[label_col].map(by_cell["nucleus_area_px"])

    valid = (
        out["dapi_mask_i"].notna()
        & out["dapi_mask_j"].notna()
        & np.isfinite(out["nuclear_dapi_mean_i"])
        & np.isfinite(out["nuclear_dapi_mean_j"])
        & (out["nuclear_dapi_mean_i"] > 0)
        & (out["nuclear_dapi_mean_j"] > 0)
    )
    dropped = int((~valid).sum())
    return out.loc[valid].reset_index(drop=True), dropped


# ============================================================================
# RIBBON SAMPLING (batched)
# ============================================================================
def distance_grid():
    """Global distance grid, shared by every pair. Keyed on an integer step index."""
    n = int(np.floor(PROFILE_HALF_LENGTH_UM / PROFILE_STEP_UM + 1e-9))
    idx = np.arange(-n, n + 1)
    return idx, idx * PROFILE_STEP_UM


def width_offsets():
    n = int(np.floor(PROFILE_HALF_WIDTH_UM / PROFILE_STEP_UM + 1e-9))
    if n == 0:
        return np.array([0.0])
    return np.arange(-n, n + 1) * PROFILE_STEP_UM


def sample_ribbons(image_cyx_float, mask_z, interfaces, pixel_size_um_yx):
    """
    Sample all interfaces of one Z slice.

    Ribbon samples that land inside a cell other than the pair are excluded from
    the width-average, and positions where too many samples are contaminated are
    dropped. Background is allowed, because a real membrane can lie just outside
    both Cellpose masks; its fraction is recorded instead.

    Returns profiles (P,C,T), n_valid (P,T), third_frac (P,T), bg_frac (P,T),
    neg_limit (P,), pos_limit (P,).
    """
    y_um, x_um = map(float, pixel_size_um_yx)
    n_pairs = len(interfaces)
    _, distances_um = distance_grid()
    offsets_um = width_offsets()
    n_ch = image_cyx_float.shape[0]
    n_t = distances_um.size
    n_w = offsets_um.size
    Y, X = image_cyx_float.shape[1:]

    profiles = np.full((n_pairs, n_ch, n_t), np.nan)
    n_valid = np.zeros((n_pairs, n_t), dtype=np.int32)
    third_frac = np.zeros((n_pairs, n_t))
    bg_frac = np.zeros((n_pairs, n_t))

    y1 = interfaces["y1"].to_numpy(float); x1 = interfaces["x1"].to_numpy(float)
    y2 = interfaces["y2"].to_numpy(float); x2 = interfaces["x2"].to_numpy(float)
    cy = interfaces["interface_y"].to_numpy(float)
    cx = interfaces["interface_x"].to_numpy(float)
    lab_i = interfaces["label_i"].to_numpy(np.int64)
    lab_j = interfaces["label_j"].to_numpy(np.int64)

    # Each side clipped against its own centroid distance, not half of D.
    neg_limit = np.minimum(PROFILE_HALF_LENGTH_UM,
                           PROFILE_LENGTH_FRACTION * interfaces["dist_to_i_um"].to_numpy(float))
    pos_limit = np.minimum(PROFILE_HALF_LENGTH_UM,
                           PROFILE_LENGTH_FRACTION * interfaces["dist_to_j_um"].to_numpy(float))

    dy = (y2 - y1) * y_um
    dx = (x2 - x1) * x_um
    norm = np.hypot(dy, dx)
    uy = dy / norm; ux = dx / norm
    py = -ux; px = uy

    mask_f = np.ascontiguousarray(mask_z, dtype=np.float32)

    for start in range(0, n_pairs, RIBBON_CHUNK):
        stop = min(start + RIBBON_CHUNK, n_pairs)
        s_ = slice(start, stop)
        m = stop - start

        d = distances_um[None, None, :]
        w = offsets_um[None, :, None]
        yy = (cy[s_][:, None, None] + d * uy[s_][:, None, None] / y_um
              + w * py[s_][:, None, None] / y_um)
        xx = (cx[s_][:, None, None] + d * ux[s_][:, None, None] / x_um
              + w * px[s_][:, None, None] / x_um)

        inbounds = (yy >= 0) & (yy <= Y - 1) & (xx >= 0) & (xx <= X - 1)
        coords = np.stack([np.clip(yy, 0, Y - 1).ravel(), np.clip(xx, 0, X - 1).ravel()])

        sampled_labels = map_coordinates(
            mask_f, coords, order=0, mode="nearest", prefilter=False
        ).reshape(m, n_w, n_t).astype(np.int64)
        is_bg = sampled_labels == 0
        is_pair = ((sampled_labels == lab_i[s_][:, None, None])
                   | (sampled_labels == lab_j[s_][:, None, None]))
        allowed = (is_bg | is_pair) & inbounds
        contaminated = inbounds & ~is_bg & ~is_pair

        n_in = inbounds.sum(axis=1)
        counts = allowed.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            third_frac[s_] = contaminated.sum(axis=1) / np.maximum(n_in, 1)
            bg_frac[s_] = (is_bg & inbounds).sum(axis=1) / np.maximum(n_in, 1)
        n_valid[s_] = counts

        for c in range(n_ch):
            sampled = map_coordinates(
                image_cyx_float[c], coords, order=1, mode="nearest", prefilter=False
            ).reshape(m, n_w, n_t)
            sampled = np.where(allowed, sampled, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = sampled.sum(axis=1) / np.maximum(counts, 1)
            profiles[s_, c, :] = np.where(counts > 0, mean, np.nan)
        del yy, xx, inbounds, coords, sampled_labels, allowed, contaminated

    within = ((distances_um[None, :] >= -neg_limit[:, None] - 1e-9)
              & (distances_um[None, :] <= pos_limit[:, None] + 1e-9))
    enough = n_valid >= int(np.ceil(MIN_VALID_WIDTH_FRACTION * n_w))
    clean = third_frac <= MAX_THIRD_CELL_FRACTION
    ok = within & enough & clean
    profiles[~ok[:, None, :].repeat(n_ch, axis=1)] = np.nan
    n_valid = np.where(ok, n_valid, 0)
    return profiles, n_valid, third_frac, bg_frac, neg_limit, pos_limit


# ============================================================================
# TABLE ASSEMBLY
# ============================================================================
def build_long_table(records):
    """records: list of dicts holding per-slice arrays. Returns the long DataFrame."""
    step_idx, distances_um = distance_grid()
    frames = []
    for rec in records:
        profiles = rec["profiles"]                 # (P, C, T)
        n_valid = rec["n_valid"]                   # (P, T)
        n_p, n_c, n_t = profiles.shape
        keep = n_valid > 0
        if not keep.any():
            continue
        pair_idx, t_idx = np.nonzero(keep)
        base = pd.DataFrame({
            "z": rec["z"],
            "label_i": rec["label_i"][pair_idx],
            "label_j": rec["label_j"][pair_idx],
            "position_index": step_idx[t_idx],
            "distance_um": distances_um[t_idx],
            "n_valid_width_samples": n_valid[keep],
            "third_cell_fraction": rec["third"][keep],
            "background_fraction": rec["bg"][keep],
            "neg_limit_um": rec["neg"][pair_idx],
            "pos_limit_um": rec["pos"][pair_idx],
            "normalization_factor": rec["norm"][pair_idx],
        })
        for c, name in enumerate(CHANNEL_ORDER):
            base[name] = profiles[pair_idx, c, t_idx]
        frames.append(base)

    if not frames:
        return pd.DataFrame()

    wide = pd.concat(frames, ignore_index=True)
    id_cols = [c for c in wide.columns if c not in CHANNEL_ORDER]
    long = wide.melt(id_vars=id_cols, value_vars=CHANNEL_ORDER,
                     var_name="channel", value_name="intensity")
    long = long[np.isfinite(long["intensity"])].copy()

    if NORMALIZATION == "pair_nuclear_dapi_mean":
        factors = long["normalization_factor"].astype(float)
        valid = np.isfinite(factors) & (factors > 0)
        if not bool(valid.all()):
            raise ValueError(
                "Invalid nuclear-DAPI normalization factor remained after pair filtering."
            )
        long["normalized_intensity"] = long["intensity"] / factors
    elif NORMALIZATION == "none":
        long["normalized_intensity"] = long["intensity"]
    else:
        raise ValueError(f"Unknown NORMALIZATION={NORMALIZATION!r}")
    return long


def build_pair_table(long, interface_frames):
    """
    One row per pair, with the mean of each channel on each side of the interface.
    No thresholding: classify downstream from these columns.
    """
    if long.empty:
        return pd.DataFrame()
    keys = ["z", "label_i", "label_j"]
    side = np.where(long["distance_um"] < 0, "negative",
                    np.where(long["distance_um"] > 0, "positive", "centre"))
    tmp = long.assign(side=side)
    tmp = tmp[tmp["side"] != "centre"]

    stats = (tmp.groupby(keys + ["channel", "side"], sort=False)["normalized_intensity"]
             .mean().unstack(["channel", "side"]))
    stats.columns = [f"{ch}_{sd}" for ch, sd in stats.columns]
    stats = stats.reset_index()

    meta = pd.concat(interface_frames, ignore_index=True) if interface_frames else pd.DataFrame()
    if not meta.empty:
        meta_cols = keys + [
            "expanded_interface_length_um", "gap_um", "centroid_distance_um",
            "n_interface_px", "dist_to_i_um", "dist_to_j_um",
            "interface_y", "interface_x",
            "dapi_mask_i", "dapi_mask_j",
            "nuclear_dapi_mean_i", "nuclear_dapi_mean_j",
            "dapi_overlap_fraction_i", "dapi_overlap_fraction_j",
            "dapi_nucleus_area_px_i", "dapi_nucleus_area_px_j",
        ]
        stats = stats.merge(meta[meta_cols], on=keys, how="left")
    return stats


def summarize(long, value_col="normalized_intensity"):
    """
    Within-image descriptive summary.

    `count` is the number of slice-local interfaces, NOT independent biological
    replicates: one physical cell pair appears in several adjacent Z slices, so
    `sem` here is descriptive spread only. For figures, aggregate these per-image
    means across images/blastoids and compute the error bar at that level.
    """
    summary = (long.groupby(["channel", "position_index"], sort=True)[value_col]
               .agg(["mean", "std", "count"]).reset_index())
    summary["distance_um"] = summary["position_index"] * PROFILE_STEP_UM
    summary["sem_within_image"] = (
        summary["std"] / np.sqrt(summary["count"].clip(lower=1))).fillna(0.0)
    summary["n_slice_local_interfaces"] = summary["count"]
    summary["sem"] = summary["sem_within_image"]   # kept for the plotting helper
    return summary


def alignment_qc(long):
    """Report the offset between the geometric zero and the QC channel's local max."""
    if ALIGNMENT_QC_CHANNEL is None or ALIGNMENT_QC_CHANNEL not in set(long["channel"]):
        return None
    sub = long[(long["channel"] == ALIGNMENT_QC_CHANNEL)
               & (long["distance_um"].abs() <= ALIGNMENT_QC_SEARCH_UM)]
    sub = sub[np.isfinite(sub["intensity"])]
    if sub.empty:
        return None
    idx = sub.groupby(["z", "label_i", "label_j"], sort=False)["intensity"].idxmax()
    offsets = sub.loc[idx, "distance_um"]
    print(
        f"  [QC] {ALIGNMENT_QC_CHANNEL} peak vs geometric zero: median {offsets.median():+.2f} µm, "
        f"IQR {offsets.quantile(0.25):+.2f} to {offsets.quantile(0.75):+.2f} µm, "
        f"n={len(offsets)} (reported only, not applied)"
    )
    return offsets


# ============================================================================
# MODEL
# ============================================================================
print("Initializing Cellpose model...")
USE_GPU = core.use_gpu()
MODEL = models.CellposeModel(gpu=USE_GPU)
print(f"Model loaded. GPU: {USE_GPU}")


def segment_stack(
    img,
    segmentation_indices,
    min_size_px,
    *,
    diameter_px,
    flow_threshold,
    cellprob_threshold,
):
    """Segment selected channels independently in each Z slice."""
    z_count = img.shape[0]
    masks = []
    for start in range(0, z_count, EVAL_CHUNK_SLICES):
        stop = min(start + EVAL_CHUNK_SLICES, z_count)
        batch = [img[z, segmentation_indices, :, :] for z in range(start, stop)]
        result = MODEL.eval(
            batch,
            channel_axis=0,
            do_3D=False,
            diameter=diameter_px,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size_px,
            progress=False,
        )
        masks.extend(np.asarray(m, dtype=np.int32) for m in result[0])
        del result, batch
        gc.collect()
    return np.stack(masks, axis=0)


# ============================================================================
# PLOTTING
# ============================================================================
def plot_segmentation_preview(img, masks, dapi_masks, name2idx, z_idx, outpath):
    dapi = percentile_norm(img[z_idx, name2idx[NUCLEAR_CHANNEL]])
    cell_overlay = label2rgb(
        masks[z_idx], image=dapi, bg_label=0, bg_color=(0, 0, 0), alpha=0.45
    )
    dapi_overlay = label2rgb(
        dapi_masks[z_idx], image=dapi, bg_label=0, bg_color=(0, 0, 0), alpha=0.45
    )
    y_size, x_size = dapi.shape
    fig, axes = plt.subplots(1, 3, figsize=image_figsize(y_size, x_size, ncols=3))
    axes[0].imshow(dapi, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"{NUCLEAR_CHANNEL}, z={z_idx}")
    axes[1].imshow(cell_overlay, interpolation="nearest")
    axes[1].set_title(f"Cell masks: {int(masks[z_idx].max())}")
    axes[2].imshow(dapi_overlay, interpolation="nearest")
    axes[2].set_title(f"DAPI masks: {int(dapi_masks[z_idx].max())}")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    savefig_close(fig, outpath)


def plot_interface_ribbons(img, masks, interfaces, name2idx, z_idx, pixel_size_um_yx, outpath):
    dapi = percentile_norm(img[z_idx, name2idx[NUCLEAR_CHANNEL]])
    overlay = label2rgb(masks[z_idx], image=dapi, bg_label=0, bg_color=(0, 0, 0), alpha=0.35)
    y_size, x_size = dapi.shape
    fig, ax = plt.subplots(figsize=image_figsize(y_size, x_size))
    ax.imshow(overlay, interpolation="nearest")

    y_um, x_um = map(float, pixel_size_um_yx)
    for _, row in interfaces.iterrows():
        dy = (row["y2"] - row["y1"]) * y_um
        dx = (row["x2"] - row["x1"]) * x_um
        norm = np.hypot(dy, dx)
        if norm < 1e-9:
            continue
        uy, ux = dy / norm, dx / norm
        py, px = -ux, uy
        neg = min(PROFILE_HALF_LENGTH_UM, PROFILE_LENGTH_FRACTION * row["dist_to_i_um"])
        pos = min(PROFILE_HALF_LENGTH_UM, PROFILE_LENGTH_FRACTION * row["dist_to_j_um"])
        corners = []
        for along, across in [(-neg, -PROFILE_HALF_WIDTH_UM), (-neg, PROFILE_HALF_WIDTH_UM),
                              (pos, PROFILE_HALF_WIDTH_UM), (pos, -PROFILE_HALF_WIDTH_UM)]:
            yy = row["interface_y"] + along * uy / y_um + across * py / y_um
            xx = row["interface_x"] + along * ux / x_um + across * px / x_um
            corners.append((xx, yy))
        ax.add_patch(Polygon(np.asarray(corners), closed=True, facecolor="yellow",
                             edgecolor="red", linewidth=0.5, alpha=0.16))
        ax.plot(row["interface_x"], row["interface_y"], marker="o", markersize=2.5,
                markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.4)

    ax.set_title(f"Sampled interface ribbons, z={z_idx} (n={len(interfaces)})")
    ax.set_axis_off()
    fig.tight_layout()
    savefig_close(fig, outpath)


def plot_average_profiles(summary, outpath):
    fig, ax = plt.subplots(figsize=(7, 5))
    for channel, group in summary.groupby("channel"):
        group = group.sort_values("distance_um")
        x = group["distance_um"].to_numpy(float)
        mean = group["mean"].to_numpy(float)
        sem = group["sem"].to_numpy(float)
        ax.plot(x, mean, label=channel)
        ax.fill_between(x, mean - sem, mean + sem, alpha=0.25)
    ax.axvline(0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xlabel("Distance from geometric interface (µm)")
    ax.set_ylabel("Normalized intensity" if NORMALIZATION != "none" else "Intensity")
    ax.set_title("Average interface profile")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig_close(fig, outpath)


def plot_pair_side_distributions(pair_stats, outpath):
    """Histograms of per-pair side means, so a threshold can be chosen by eye."""
    channels = [c for c in CHANNEL_ORDER if f"{c}_negative" in pair_stats.columns]
    if not channels:
        return
    fig, axes = plt.subplots(1, len(channels), figsize=(4.2 * len(channels), 4.0), squeeze=False)
    for ax, channel in zip(axes[0], channels):
        both = pd.concat([pair_stats[f"{channel}_negative"], pair_stats[f"{channel}_positive"]])
        both = both.replace([np.inf, -np.inf], np.nan).dropna()
        if both.empty:
            ax.set_axis_off()
            continue
        ax.hist(both, bins=60, color="steelblue")
        ax.set_title(channel)
        ax.set_xlabel("per-side mean")
    axes[0][0].set_ylabel("pair sides")
    fig.suptitle("Per-pair side intensities (choose thresholds downstream)")
    fig.tight_layout()
    savefig_close(fig, outpath)


def plot_qc_offsets(offsets, outpath):
    if offsets is None or len(offsets) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(offsets, bins=41, color="darkorange")
    ax.axvline(0, linestyle="--", linewidth=0.8, color="gray")
    ax.axvline(float(np.median(offsets)), linestyle="-", linewidth=1.2, color="black",
               label=f"median {np.median(offsets):+.2f} µm")
    ax.set_xlabel(f"{ALIGNMENT_QC_CHANNEL} peak offset from geometric zero (µm)")
    ax.set_ylabel("pairs")
    ax.set_title("Alignment QC (measured, not applied)")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig_close(fig, outpath)


# ============================================================================
# MAIN FILE PROCESSING
# ============================================================================
def process_single_file(img_path, sample_name):
    sample_out_dir = os.path.join(OUT_DIR, sample_name)
    os.makedirs(sample_out_dir, exist_ok=True)
    profile_csv = os.path.join(sample_out_dir, f"{sample_name}_profiles.csv")
    pair_csv = os.path.join(sample_out_dir, f"{sample_name}_pair_stats.csv")
    summary_csv = os.path.join(sample_out_dir, f"{sample_name}_summary.csv")
    # Written only after every output succeeds, so a crash mid-file does not
    # cause the next run to skip an incompletely processed sample.
    done_marker = os.path.join(sample_out_dir, f"{sample_name}.complete")

    if SKIP_EXISTING and os.path.exists(done_marker):
        print(f"  [Skip] Completed previously: {done_marker}")
        return

    check_voxel_size_against_file(img_path, VOXEL_SIZE_UM)
    pixel_size_um_yx = (float(VOXEL_SIZE_UM[1]), float(VOXEL_SIZE_UM[2]))
    y_um, x_um = pixel_size_um_yx

    img = read_tiff_zcxy(img_path)
    z_count, c_count, y_size, x_size = img.shape
    name2idx = channel_map_for_image(img)
    for name in SEGMENTATION_CHANNELS + [NUCLEAR_CHANNEL]:
        if name not in name2idx:
            raise ValueError(f"Channel {name!r} not in CHANNEL_ORDER={CHANNEL_ORDER}")

    min_size_px = max(1, int(np.ceil(MIN_OBJECT_AREA_UM2 / (y_um * x_um))))
    dapi_min_size_px = max(1, int(np.ceil(MIN_DAPI_MASK_AREA_UM2 / (y_um * x_um))))
    print(f"  > Z={z_count}, C={c_count}, Y={y_size}, X={x_size}; "
          f"pixel {y_um:.4f} um; cell-mask min_size={min_size_px} px "
          f"(= {MIN_OBJECT_AREA_UM2} um2); DAPI-mask min_size={dapi_min_size_px} px "
          f"(= {MIN_DAPI_MASK_AREA_UM2} um2)")

    segmentation_indices = [name2idx[n] for n in SEGMENTATION_CHANNELS]
    print(f"  > Segmenting cell masks in {z_count} slices using {SEGMENTATION_CHANNELS}...")
    masks = segment_stack(
        img,
        segmentation_indices,
        min_size_px,
        diameter_px=DIAMETER_PX,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
    )
    print(f"  > {int(sum(m.max() for m in masks))} slice-local cell masks across "
          f"{int(sum(m.max() > 0 for m in masks))}/{z_count} slices")

    dapi_index = name2idx[NUCLEAR_CHANNEL]
    print(f"  > Segmenting DAPI masks in {z_count} slices using [{NUCLEAR_CHANNEL!r}]...")
    dapi_masks = segment_stack(
        img,
        [dapi_index],
        dapi_min_size_px,
        diameter_px=DAPI_DIAMETER_PX,
        flow_threshold=DAPI_FLOW_THRESHOLD,
        cellprob_threshold=DAPI_CELLPROB_THRESHOLD,
    )
    print(f"  > {int(sum(m.max() for m in dapi_masks))} slice-local DAPI masks across "
          f"{int(sum(m.max() > 0 for m in dapi_masks))}/{z_count} slices")

    # Report the object-size distribution so MIN_OBJECT_AREA_UM2 can be checked
    # against real cells rather than assumed.
    areas = []
    for z in range(z_count):
        if masks[z].max():
            _, counts = np.unique(masks[z][masks[z] > 0], return_counts=True)
            areas.append(counts)
    if areas:
        areas_um2 = np.concatenate(areas) * y_um * x_um
        print(f"  > Object area um2: p5={np.percentile(areas_um2,5):.0f} "
              f"median={np.median(areas_um2):.0f} p95={np.percentile(areas_um2,95):.0f} "
              f"(cutoff {MIN_OBJECT_AREA_UM2})")

    # Report the gap distribution before filtering, so MAX_PAIR_GAP_UM can be
    # set from the data rather than assumed. A large drop fraction usually means
    # the Cellpose masks sit well inside the real cell outlines.
    probe_z = [z for z in range(z_count) if masks[z].max() > 0][:3]
    probe_gaps = []
    for z in probe_z:
        expanded, dist_um = physically_expand_labels(
            masks[z], NEIGHBOR_EXPANSION_UM, pixel_size_um_yx)
        for a, b, da, db in ((expanded[:, :-1], expanded[:, 1:],
                              dist_um[:, :-1], dist_um[:, 1:]),
                             (expanded[:-1, :], expanded[1:, :],
                              dist_um[:-1, :], dist_um[1:, :])):
            v = (a != b) & (a > 0) & (b > 0)
            if v.any():
                probe_gaps.append(da[v] + db[v])
    if probe_gaps:
        g = np.concatenate(probe_gaps)
        print(f"  > Mask-to-mask gap at candidate interfaces um: p10={np.percentile(g,10):.2f} "
              f"median={np.median(g):.2f} p90={np.percentile(g,90):.2f} "
              f"(MAX_PAIR_GAP_UM={MAX_PAIR_GAP_UM})")

    print("  > Finding interfaces, assigning DAPI masks, and sampling ribbons...")
    records, interface_frames = [], []
    interface_counts = {}
    n_candidate_pairs = 0
    n_geometry_pairs = 0
    n_pairs_missing_dapi = 0
    n_dapi_masks_total = 0
    n_dapi_cells_assigned = 0
    n_duplicate_dapi_assignments = 0

    for z in range(z_count):
        if masks[z].max() == 0:
            continue

        interfaces, n_candidates = find_interfaces(masks[z], pixel_size_um_yx)
        n_candidate_pairs += n_candidates
        n_geometry_pairs += len(interfaces)
        if interfaces.empty:
            continue

        assignments, dapi_stats = map_dapi_masks_to_cells(
            masks[z],
            dapi_masks[z],
            img[z, dapi_index],
        )
        n_dapi_masks_total += dapi_stats["n_dapi_masks"]
        n_dapi_cells_assigned += dapi_stats["n_cells_assigned"]
        n_duplicate_dapi_assignments += dapi_stats["n_duplicate_cell_assignments"]

        interfaces, n_dropped_dapi = attach_nuclear_dapi_to_interfaces(
            interfaces, assignments
        )
        n_pairs_missing_dapi += n_dropped_dapi
        if interfaces.empty:
            continue

        # Both cells now have valid DAPI masks. The normalization factor is the
        # arithmetic mean of the two nuclear-mask mean DAPI intensities.
        norm = 0.5 * (
            interfaces["nuclear_dapi_mean_i"].to_numpy(float)
            + interfaces["nuclear_dapi_mean_j"].to_numpy(float)
        )
        if not np.all(np.isfinite(norm) & (norm > 0)):
            raise ValueError("Invalid DAPI normalization factor after mandatory mask filtering.")

        interfaces.insert(0, "z", z)
        interface_frames.append(interfaces)
        interface_counts[z] = len(interfaces)

        slice_float = np.ascontiguousarray(img[z], dtype=np.float32)
        profiles, n_valid, third, bg, neg, pos = sample_ribbons(
            slice_float, masks[z], interfaces, pixel_size_um_yx)
        del slice_float

        records.append(dict(
            z=z,
            label_i=interfaces["label_i"].to_numpy(int),
            label_j=interfaces["label_j"].to_numpy(int),
            profiles=profiles, n_valid=n_valid, third=third, bg=bg,
            neg=neg, pos=pos, norm=norm,
        ))

    total_pairs = sum(len(f) for f in interface_frames)
    print(
        f"  > DAPI assignment: {n_dapi_cells_assigned} cell masks assigned from "
        f"{n_dapi_masks_total} DAPI masks; {n_pairs_missing_dapi} geometric pairs "
        f"removed because one or both cells lacked a valid DAPI mask"
    )
    if n_duplicate_dapi_assignments:
        print(
            f"  [QC] {n_duplicate_dapi_assignments} DAPI-mask assignments involved cells "
            "with multiple candidate nuclei; the strongest-overlap nucleus was retained"
        )
    print(
        f"  > {total_pairs} interfaces retained after geometry and mandatory DAPI-mask "
        f"filtering ({n_geometry_pairs} geometric; {n_candidate_pairs} candidates)"
    )
    if n_candidate_pairs and total_pairs < 0.5 * n_candidate_pairs:
        print(
            f"  [WARNING] {100*(1-total_pairs/n_candidate_pairs):.0f}% of candidate pairs "
            "were filtered out. Check the cell masks, DAPI masks, "
            f"MIN_DAPI_CELL_OVERLAP_FRACTION ({MIN_DAPI_CELL_OVERLAP_FRACTION}), "
            f"MAX_PAIR_GAP_UM ({MAX_PAIR_GAP_UM} um), and "
            f"MIN_INTERFACE_LENGTH_UM ({MIN_INTERFACE_LENGTH_UM} um)."
        )

    z_preview = choose_preview_z(z_count, interface_counts)
    plot_segmentation_preview(
        img, masks, dapi_masks, name2idx, z_preview,
        os.path.join(sample_out_dir, f"{sample_name}_segmentation_preview_z{z_preview}.png"))
    preview_interfaces = next(
        (f for f in interface_frames if int(f["z"].iloc[0]) == z_preview), None)
    plot_interface_ribbons(
        img, masks,
        preview_interfaces if preview_interfaces is not None else pd.DataFrame(
            columns=["y1", "x1", "y2", "x2", "interface_y", "interface_x",
                     "dist_to_i_um", "dist_to_j_um"]),
        name2idx, z_preview, pixel_size_um_yx,
        os.path.join(sample_out_dir, f"{sample_name}_interface_ribbons_z{z_preview}.png"))

    long = build_long_table(records)
    if long.empty:
        pd.DataFrame().to_csv(pair_csv, index=False)
        with open(done_marker, "w") as handle:
            handle.write("complete: no usable profiles\n")
        print(f"  > No usable profiles. Wrote empty {pair_csv}")
        return

    long.insert(0, "sample", sample_name)
    long.insert(0, "file_name", os.path.basename(img_path))
    # Orientation convention: negative distances point toward label_i, positive
    # toward label_j. Label numbers carry no biological meaning; use the per-pair
    # side columns in *_pair_stats.csv to assign identity downstream.
    long["negative_side_label"] = long["label_i"]
    long["positive_side_label"] = long["label_j"]

    offsets = alignment_qc(long)
    plot_qc_offsets(offsets, os.path.join(sample_out_dir, f"{sample_name}_alignment_qc.png"))

    pair_stats = build_pair_table(long, interface_frames)
    if not pair_stats.empty:
        pair_stats.insert(0, "sample", sample_name)
        pair_stats.insert(0, "file_name", os.path.basename(img_path))
    pair_stats.to_csv(pair_csv, index=False)
    print(f"  > Saved pair statistics: {pair_csv} ({len(pair_stats)} pairs)")

    summary = summarize(long)
    if WRITE_SUMMARY_TABLE:
        summary.insert(0, "sample", sample_name)
        summary.to_csv(summary_csv, index=False)
        print(f"  > Saved summary: {summary_csv}")
    plot_average_profiles(summary, os.path.join(sample_out_dir, f"{sample_name}_average_profile.png"))
    plot_pair_side_distributions(
        pair_stats, os.path.join(sample_out_dir, f"{sample_name}_pair_side_distributions.png"))

    if WRITE_PROFILE_TABLE:
        long.to_csv(profile_csv, index=False)
        print(f"  > Saved profiles: {profile_csv} ({len(long)} rows)")

    with open(done_marker, "w") as handle:
        handle.write("complete\n")


def main():
    start = time.time()
    files = sorted(
        glob.glob(os.path.join(ROOT_DIR, "*.tif")) + glob.glob(os.path.join(ROOT_DIR, "*.tiff")),
        key=natsort.natsort_key,
    )
    print(f"Found {len(files)} TIFF files in {ROOT_DIR}")

    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "-1"))
    if task_id >= 0:
        if task_id >= len(files):
            print(f"[array] task {task_id} is outside 0..{len(files)-1}; exiting.")
            return
        files = [files[task_id]]
        print(f"[array] task {task_id}: {files[0]}")

    for index, path in enumerate(files, start=1):
        sample = sample_from_filename(os.path.basename(path))
        print(f"\n[{index}/{len(files)}] Processing: {sample}")
        try:
            process_single_file(path, sample)
        except Exception as exc:
            print(f"!!! Error processing {sample}: {exc}")
            traceback.print_exc()
        finally:
            plt.close("all")
            gc.collect()

    print(f"\nAll done in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
