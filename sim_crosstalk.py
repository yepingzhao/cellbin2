from __future__ import annotations

import argparse
import errno
import json
import os
import site
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import to_rgb
from scipy import sparse
from shapely import wkt
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.strtree import STRtree
from skimage.measure import find_contours, label, regionprops_table

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent
DEFAULT_MC_GEM = REPO_ROOT / "data" / "Y40178MC" / "Y40178MC.gem"
DEFAULT_P5_GEM = REPO_ROOT / "data" / "Y40178P5" / "Y40178P5.gem"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "sim_crosstalk"
DEFAULT_CELLBIN2_TEMPLATE = REPO_ROOT / "cellbin2" / "config" / "demos" / "Stereocell_analysis.json"
DEFAULT_CELLBIN2_ENTRY = REPO_ROOT / "cellbin2" / "cellbin_pipeline.py"
DEFAULT_CELLBIN2_PYTHON = Path(sys.executable)
GEM_COLUMNS = ["geneID", "x", "y", "MIDCount", "ExonCount"]
CELL_COLUMNS = [
    "cell_label",
    "source",
    "pixel_label",
    "area",
    "bbox_min_x",
    "bbox_min_y",
    "bbox_max_x",
    "bbox_max_y",
    "centroid_x",
    "centroid_y",
    "contour_wkt",
]
DEFAULT_TILE_GRID_SIZE = 20
DEFAULT_TILE_ROW = 10
DEFAULT_TILE_COL = 10
DEFAULT_SPATIAL_TILE_FILENAME = f"spatial_tile_r{DEFAULT_TILE_ROW}_c{DEFAULT_TILE_COL}.png"
PANEL_COLORS = {
    "MC": "#1b9e77",
    "P5": "#d95f02",
    "mixed_before": "#9e9e9e",
    "mixed_after": "#4c78a8",
    "mapped_mixed": "#2a9d8f",
}


def log_step(message: str) -> None:
    print(message, flush=True)


def read_gem_header(path: Path) -> list[str]:
    header_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header_lines.append(line.rstrip("\n"))
    return header_lines


def iter_gem_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        sep="\t",
        comment="#",
        chunksize=chunk_size,
        dtype={
            "geneID": "string",
            "x": np.int32,
            "y": np.int32,
            "MIDCount": np.int32,
            "ExonCount": np.int32,
        },
    )


def build_mixed_gem(mc_gem: Path, p5_gem: Path, output_path: Path, chunk_size: int = 1_000_000) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = read_gem_header(mc_gem)
    row_count = 0
    total_mid = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for line in header_lines:
            handle.write(f"{line}\n")
        wrote_header = False
        for gem_path in (mc_gem, p5_gem):
            for chunk in iter_gem_chunks(gem_path, chunk_size):
                if chunk.empty:
                    continue
                row_count += int(len(chunk))
                total_mid += int(chunk["MIDCount"].sum())
                chunk.to_csv(
                    handle,
                    sep="\t",
                    index=False,
                    header=not wrote_header,
                    columns=GEM_COLUMNS,
                )
                wrote_header = True
        if not wrote_header:
            pd.DataFrame(columns=GEM_COLUMNS).to_csv(handle, sep="\t", index=False)

    return {
        "mixed_row_count": row_count,
        "mixed_mid_total": total_mid,
    }


def convert_gem_to_gef(gem_path: Path, gef_path: Path) -> Path:
    from cellbin2.matrix.matrix import gem_to_gef

    gef_path.parent.mkdir(parents=True, exist_ok=True)
    gem_to_gef(str(gem_path), str(gef_path))
    if not gef_path.exists():
        raise FileNotFoundError(f"failed to create GEF: {gef_path}")
    return gef_path


def remove_tree_with_retries(path: Path, retries: int = 3, delay_seconds: float = 0.2) -> None:
    last_error: OSError | None = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            last_error = exc
            if exc.errno != errno.ENOTEMPTY or attempt == retries - 1:
                raise
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error


def ensure_output_dir(output_dir: Path, force: bool) -> None:
    if force and output_dir.exists():
        remove_tree_with_retries(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_cellbin2_config(
    sample_name: str,
    gem_path: Path,
    template_path: Path,
    config_path: Path,
) -> Path:
    config = json.loads(template_path.read_text(encoding="utf-8"))
    config["image_process"]["0"]["file_path"] = str(gem_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def discover_onnxruntime_gpu_lib_paths() -> list[str]:
    relative_paths = [
        "nvidia/cudnn/lib",
        "nvidia/cublas/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/cuda_nvrtc/lib",
        "nvidia/cufft/lib",
        "nvidia/curand/lib",
        "nvidia/cusolver/lib",
        "nvidia/cusparse/lib",
        "nvidia/nvjitlink/lib",
        "nvidia/nccl/lib",
        "nvidia/nvtx/lib",
        "nvidia/cuda_cupti/lib",
    ]
    roots = site.getsitepackages() + [site.getusersitepackages()]
    found_paths: list[str] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for relative_path in relative_paths:
            lib_path = root_path / relative_path
            if lib_path.exists():
                found_paths.append(str(lib_path))
    return found_paths


def build_cellbin2_env() -> dict[str, str]:
    env = os.environ.copy()
    gpu_libs = discover_onnxruntime_gpu_lib_paths()
    if gpu_libs:
        existing = env.get("LD_LIBRARY_PATH", "")
        ordered_paths = gpu_libs + ([existing] if existing else [])
        env["LD_LIBRARY_PATH"] = ":".join(path for path in ordered_paths if path)
    return env


def run_cellbin2(
    sample_name: str,
    config_path: Path,
    run_output_dir: Path,
    cellbin2_python: Path,
    cellbin2_entry: Path,
) -> dict[str, str]:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(cellbin2_python),
        str(cellbin2_entry),
        "-c",
        sample_name,
        "-p",
        str(config_path),
        "-o",
        str(run_output_dir),
    ]
    completed = subprocess.run(command, check=False, env=build_cellbin2_env())
    cell_mask_path = run_output_dir / f"{sample_name}_cell_mask.tif"
    if completed.returncode != 0 and not cell_mask_path.exists():
        raise subprocess.CalledProcessError(completed.returncode, command)
    if not cell_mask_path.exists():
        raise FileNotFoundError(f"missing CellBin2 output mask: {cell_mask_path}")
    return {
        "cell_mask": str(cell_mask_path),
        "returncode": str(completed.returncode),
    }


def get_run_output(sample_name: str, run_output_dir: Path) -> dict[str, str]:
    cell_mask_path = run_output_dir / f"{sample_name}_cell_mask.tif"
    if not cell_mask_path.exists():
        raise FileNotFoundError(f"missing existing CellBin2 output mask: {cell_mask_path}")
    return {
        "cell_mask": str(cell_mask_path),
        "returncode": "existing",
    }


def load_final_cell_labels(mask_path: Path) -> np.ndarray:
    mask = tifffile.imread(mask_path)
    if np.issubdtype(mask.dtype, np.integer):
        label_image = mask.astype(np.int32)
    else:
        label_image = mask.astype(np.int32)
    if label_image.max() <= 1:
        label_image = label(label_image > 0, connectivity=1).astype(np.int32)
    return label_image


def contour_to_polygon(points: np.ndarray) -> Polygon:
    if len(points) < 3:
        raise ValueError("contour requires at least 3 points")
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise ValueError("contour produced empty polygon")
    return polygon


def extract_cell_polygons(label_image: np.ndarray, source_name: str) -> pd.DataFrame:
    if int(label_image.max()) == 0:
        return pd.DataFrame(columns=CELL_COLUMNS)

    props = regionprops_table(label_image, properties=("label", "area", "bbox", "centroid"))
    rows: list[dict[str, object]] = []
    for idx in range(len(props["label"])):
        pixel_label = int(props["label"][idx])
        min_row = int(props["bbox-0"][idx])
        min_col = int(props["bbox-1"][idx])
        max_row = int(props["bbox-2"][idx]) - 1
        max_col = int(props["bbox-3"][idx]) - 1
        region_mask = label_image[min_row : max_row + 1, min_col : max_col + 1] == pixel_label
        if region_mask.shape[0] < 2 or region_mask.shape[1] < 2:
            polygon = box(min_col, min_row, max_col + 1, max_row + 1)
        else:
            contours = find_contours(region_mask.astype(np.uint8), level=0.5)
            if contours:
                contour = max(contours, key=len)
                contour_xy = np.column_stack((contour[:, 1] + min_col, contour[:, 0] + min_row))
                if len(contour_xy) < 3:
                    polygon = box(min_col, min_row, max_col + 1, max_row + 1)
                else:
                    try:
                        polygon = contour_to_polygon(contour_xy)
                    except ValueError:
                        polygon = box(min_col, min_row, max_col + 1, max_row + 1)
            else:
                polygon = box(min_col, min_row, max_col + 1, max_row + 1)
        rows.append(
            {
                "cell_label": f"{source_name}_{pixel_label}",
                "source": source_name,
                "pixel_label": pixel_label,
                "area": int(props["area"][idx]),
                "bbox_min_x": min_col,
                "bbox_min_y": min_row,
                "bbox_max_x": max_col,
                "bbox_max_y": max_row,
                "centroid_x": float(props["centroid-1"][idx]),
                "centroid_y": float(props["centroid-0"][idx]),
                "contour_wkt": polygon.wkt,
            }
        )
    return pd.DataFrame(rows, columns=CELL_COLUMNS)


def assign_molecules_to_cells(
    gem_path: Path,
    label_image: np.ndarray,
    source_name: str,
    chunk_size: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    height, width = label_image.shape
    for chunk in iter_gem_chunks(gem_path, chunk_size):
        if chunk.empty:
            continue
        rel_x = chunk["x"].to_numpy(dtype=np.int64)
        rel_y = chunk["y"].to_numpy(dtype=np.int64)
        in_bounds = (
            (rel_x >= 0)
            & (rel_x < width)
            & (rel_y >= 0)
            & (rel_y < height)
        )
        pixel_labels = np.zeros(len(chunk), dtype=np.int32)
        pixel_labels[in_bounds] = label_image[rel_y[in_bounds], rel_x[in_bounds]]
        assigned = chunk.loc[pixel_labels > 0, GEM_COLUMNS].copy()
        if assigned.empty:
            continue
        assigned_labels = pixel_labels[pixel_labels > 0]
        assigned["cell_label"] = [f"{source_name}_{value}" for value in assigned_labels]
        assigned["pixel_label"] = assigned_labels
        assigned["source"] = source_name
        rows.append(assigned)
    if not rows:
        return pd.DataFrame(columns=GEM_COLUMNS + ["cell_label", "pixel_label", "source"])
    return pd.concat(rows, ignore_index=True)


def build_spatial_index(cell_df: pd.DataFrame) -> tuple[STRtree, list[Polygon], dict[int, str]]:
    polygons = [wkt.loads(value) for value in cell_df["contour_wkt"].tolist()]
    tree = STRtree(polygons)
    index_to_label = {
        index: str(cell_df.iloc[index]["cell_label"]) for index in range(len(cell_df))
    }
    return tree, polygons, index_to_label


def _collect_overlap_candidates(
    mixed_cells: pd.DataFrame,
    source_cells: pd.DataFrame,
    source_name: str,
    iou_threshold: float,
) -> list[dict[str, object]]:
    if mixed_cells.empty or source_cells.empty:
        return []

    tree, polygons, index_to_label = build_spatial_index(source_cells)
    candidates: list[dict[str, object]] = []
    for mixed_index, mixed_row in mixed_cells.iterrows():
        mixed_polygon = wkt.loads(str(mixed_row["contour_wkt"]))
        candidate_indices = tree.query(mixed_polygon)
        for candidate_index in candidate_indices.tolist():
            source_polygon = polygons[int(candidate_index)]
            intersection_area = float(mixed_polygon.intersection(source_polygon).area)
            if intersection_area <= 0.0:
                continue
            union_area = float(mixed_polygon.union(source_polygon).area)
            iou = intersection_area / union_area if union_area > 0.0 else 0.0
            if iou < iou_threshold:
                continue
            candidates.append(
                {
                    "mixed_cell_label": str(mixed_row["cell_label"]),
                    "mixed_source": str(mixed_row["source"]),
                    "source_dataset": source_name,
                    "source_cell_label": index_to_label[int(candidate_index)],
                    "iou": iou,
                    "intersection_area": intersection_area,
                    "union_area": union_area,
                }
            )
    return candidates


def map_mixed_cells(
    mixed_cells: pd.DataFrame,
    mc_cells: pd.DataFrame,
    p5_cells: pd.DataFrame,
    iou_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _collect_overlap_candidates(mixed_cells, mc_cells, "MC", iou_threshold)
    candidates.extend(_collect_overlap_candidates(mixed_cells, p5_cells, "P5", iou_threshold))
    candidates_df = pd.DataFrame(
        candidates,
        columns=[
            "mixed_cell_label",
            "mixed_source",
            "source_dataset",
            "source_cell_label",
            "iou",
            "intersection_area",
            "union_area",
        ],
    )

    mapping_rows: list[dict[str, object]] = []
    for mixed_label in mixed_cells["cell_label"].tolist():
        mixed_candidates = candidates_df.loc[candidates_df["mixed_cell_label"] == mixed_label].copy()
        source_datasets = sorted(mixed_candidates["source_dataset"].unique().tolist()) if not mixed_candidates.empty else []
        is_doublet = len(source_datasets) > 1
        if mixed_candidates.empty:
            mapping_rows.append(
                {
                    "mixed_cell_label": mixed_label,
                    "mapped_source_dataset": "",
                    "mapped_source_label": "",
                    "mapped_iou": 0.0,
                    "mapped_intersection_area": 0.0,
                    "is_doublet": False,
                    "doublet_reason": "",
                }
            )
            continue
        mixed_candidates = mixed_candidates.sort_values(
            by=["iou", "intersection_area", "source_cell_label"],
            ascending=[False, False, True],
        )
        best = mixed_candidates.iloc[0]
        mapping_rows.append(
            {
                "mixed_cell_label": mixed_label,
                "mapped_source_dataset": str(best["source_dataset"]),
                "mapped_source_label": str(best["source_cell_label"]),
                "mapped_iou": float(best["iou"]),
                "mapped_intersection_area": float(best["intersection_area"]),
                "is_doublet": is_doublet,
                "doublet_reason": "cross_source_overlap" if is_doublet else "",
            }
        )
    mapping_df = pd.DataFrame(mapping_rows)
    return candidates_df, mapping_df


def filter_mixed_doublets(molecules: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    drop_labels = set(mapping_df.loc[mapping_df["is_doublet"], "mixed_cell_label"].tolist())
    if not drop_labels:
        return molecules.copy()
    return molecules.loc[~molecules["cell_label"].isin(drop_labels)].copy()


def compute_chip_tile_bounds(
    mask_shape: tuple[int, int],
    grid_size: int = DEFAULT_TILE_GRID_SIZE,
    tile_row: int = DEFAULT_TILE_ROW,
    tile_col: int = DEFAULT_TILE_COL,
) -> tuple[int, int, int, int]:
    height, width = mask_shape
    if grid_size < 1:
        raise ValueError("grid_size must be >= 1")
    if tile_row < 1 or tile_row > grid_size or tile_col < 1 or tile_col > grid_size:
        raise ValueError("tile_row and tile_col must be within 1..grid_size")

    x_edges = np.linspace(0, width, grid_size + 1, dtype=int)
    y_edges = np.linspace(0, height, grid_size + 1, dtype=int)
    x0 = int(x_edges[tile_col - 1])
    x1 = int(x_edges[tile_col])
    y0 = int(y_edges[tile_row - 1])
    y1 = int(y_edges[tile_row])

    if width > 0 and x1 <= x0:
        x0 = min(x0, width - 1)
        x1 = min(width, x0 + 1)
    if height > 0 and y1 <= y0:
        y0 = min(y0, height - 1)
        y1 = min(height, y0 + 1)
    return x0, y0, x1, y1


def select_cells_in_view(cell_df: pd.DataFrame, view_bounds: tuple[int, int, int, int]) -> pd.DataFrame:
    if cell_df.empty:
        return cell_df.copy()

    x0, y0, x1, y1 = view_bounds
    bbox_hits = cell_df.loc[
        (cell_df["bbox_max_x"] >= x0)
        & (cell_df["bbox_min_x"] <= x1)
        & (cell_df["bbox_max_y"] >= y0)
        & (cell_df["bbox_min_y"] <= y1)
    ].copy()
    if bbox_hits.empty:
        return bbox_hits

    view_polygon = box(x0, y0, x1, y1)
    intersects = bbox_hits["contour_wkt"].map(lambda value: wkt.loads(str(value)).intersects(view_polygon))
    return bbox_hits.loc[intersects].copy()


def _extract_tile_labels(label_image: np.ndarray, view_bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, set[int]]:
    tile = _extract_view_tile(label_image, view_bounds)
    labels = set(np.unique(tile).tolist())
    labels.discard(0)
    return tile, {int(value) for value in labels}


def _extract_view_tile(
    label_image: np.ndarray,
    view_bounds: tuple[int, int, int, int],
    *,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    x0, y0, x1, y1 = view_bounds
    view_height = max(0, y1 - y0)
    view_width = max(0, x1 - x0)
    if target_shape is None:
        target_shape = (view_height, view_width)

    tile = np.zeros(target_shape, dtype=label_image.dtype)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(label_image.shape[1], x1)
    src_y1 = min(label_image.shape[0], y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return tile

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    tile[dst_y0:dst_y1, dst_x0:dst_x1] = label_image[src_y0:src_y1, src_x0:src_x1]
    return tile


def _filter_tile_by_labels(tile: np.ndarray, allowed_labels: set[int] | None) -> np.ndarray:
    if allowed_labels is None:
        return tile.copy()
    if not allowed_labels:
        return np.zeros_like(tile)
    filtered = tile.copy()
    filtered[~np.isin(filtered, list(allowed_labels))] = 0
    return filtered


def _lookup_pixel_labels(cell_df: pd.DataFrame, cell_labels: set[str]) -> set[int]:
    if cell_df.empty or not cell_labels:
        return set()
    matched = cell_df.loc[cell_df["cell_label"].isin(cell_labels), "pixel_label"].tolist()
    return {int(value) for value in matched}


def _render_single_mask_panel(
    ax: plt.Axes,
    label_tile: np.ndarray,
    view_bounds: tuple[int, int, int, int],
    title: str,
    color: str,
) -> None:
    x0, y0, x1, y1 = view_bounds
    rgb = np.ones((label_tile.shape[0], label_tile.shape[1], 3), dtype=np.float32)
    rgb[label_tile > 0] = np.asarray(to_rgb(color), dtype=np.float32)
    ax.imshow(rgb, extent=(x0, x1, y1, y0), interpolation="nearest")
    ax.set_title(f"{title}\n{len(np.unique(label_tile[label_tile > 0]))} cells", fontsize=11)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)


def _render_source_mask_panel(
    ax: plt.Axes,
    mc_tile: np.ndarray,
    p5_tile: np.ndarray,
    view_bounds: tuple[int, int, int, int],
    title: str,
) -> None:
    x0, y0, x1, y1 = view_bounds
    rgb = np.ones((mc_tile.shape[0], mc_tile.shape[1], 3), dtype=np.float32)
    mc_color = np.asarray(to_rgb(PANEL_COLORS["MC"]), dtype=np.float32)
    p5_color = np.asarray(to_rgb(PANEL_COLORS["P5"]), dtype=np.float32)
    mc_mask = mc_tile > 0
    p5_mask = p5_tile > 0
    rgb[mc_mask] = mc_color
    rgb[p5_mask] = p5_color
    overlap = mc_mask & p5_mask
    if np.any(overlap):
        rgb[overlap] = (mc_color + p5_color) / 2.0

    ax.imshow(rgb, extent=(x0, x1, y1, y0), interpolation="nearest")
    cell_count = len(np.unique(mc_tile[mc_tile > 0])) + len(np.unique(p5_tile[p5_tile > 0]))
    ax.set_title(f"{title}\n{cell_count} cells", fontsize=11)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)


def write_spatial_visualization(
    output_path: Path,
    mc_cells: pd.DataFrame,
    p5_cells: pd.DataFrame,
    mixed_cells: pd.DataFrame,
    mapping_df: pd.DataFrame,
    mc_labels: np.ndarray,
    p5_labels: np.ndarray,
    mixed_labels: np.ndarray,
    *,
    grid_size: int = DEFAULT_TILE_GRID_SIZE,
    tile_row: int = DEFAULT_TILE_ROW,
    tile_col: int = DEFAULT_TILE_COL,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mc_view = compute_chip_tile_bounds(mc_labels.shape, grid_size=grid_size, tile_row=tile_row, tile_col=tile_col)
    p5_view = compute_chip_tile_bounds(p5_labels.shape, grid_size=grid_size, tile_row=tile_row, tile_col=tile_col)
    mixed_view = compute_chip_tile_bounds(mixed_labels.shape, grid_size=grid_size, tile_row=tile_row, tile_col=tile_col)

    doublet_labels = set(mapping_df.loc[mapping_df["is_doublet"], "mixed_cell_label"].tolist())
    mapped_rows = mapping_df.loc[
            (~mapping_df["is_doublet"])
            & mapping_df["mapped_source_dataset"].ne("")
            & mapping_df["mapped_source_label"].ne(""),
    ].copy()
    mapped_labels = set(mapped_rows["mixed_cell_label"].tolist())

    mc_tile, _ = _extract_tile_labels(mc_labels, mc_view)
    p5_tile, _ = _extract_tile_labels(p5_labels, p5_view)
    mixed_tile, mixed_tile_pixel_labels = _extract_tile_labels(mixed_labels, mixed_view)
    mixed_tile_label_names = {f"mixed_{value}" for value in mixed_tile_pixel_labels}

    doublet_pixel_labels = _lookup_pixel_labels(mixed_cells, doublet_labels)
    mapped_pixel_labels = _lookup_pixel_labels(mixed_cells, mapped_labels)
    mixed_after_tile = _filter_tile_by_labels(mixed_tile, mixed_tile_pixel_labels - doublet_pixel_labels)
    mapped_mixed_tile = _filter_tile_by_labels(mixed_tile, mixed_tile_pixel_labels & mapped_pixel_labels)

    selected_mapped_rows = mapped_rows.loc[mapped_rows["mixed_cell_label"].isin(mixed_tile_label_names)].copy()
    mc_source_labels = set(
        selected_mapped_rows.loc[selected_mapped_rows["mapped_source_dataset"] == "MC", "mapped_source_label"].tolist()
    )
    p5_source_labels = set(
        selected_mapped_rows.loc[selected_mapped_rows["mapped_source_dataset"] == "P5", "mapped_source_label"].tolist()
    )
    mc_source_pixels = _lookup_pixel_labels(mc_cells, mc_source_labels)
    p5_source_pixels = _lookup_pixel_labels(p5_cells, p5_source_labels)
    source_target_shape = mixed_tile.shape
    mc_source_tile = _filter_tile_by_labels(
        _extract_view_tile(mc_labels, mixed_view, target_shape=source_target_shape),
        mc_source_pixels,
    )
    p5_source_tile = _filter_tile_by_labels(
        _extract_view_tile(p5_labels, mixed_view, target_shape=source_target_shape),
        p5_source_pixels,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    _render_single_mask_panel(axes[0, 0], mc_tile, mc_view, f"MC tile ({tile_row},{tile_col})", PANEL_COLORS["MC"])
    _render_single_mask_panel(axes[0, 1], p5_tile, p5_view, f"P5 tile ({tile_row},{tile_col})", PANEL_COLORS["P5"])
    _render_single_mask_panel(axes[0, 2], mixed_tile, mixed_view, f"mixed before doublet tile ({tile_row},{tile_col})", PANEL_COLORS["mixed_before"])
    _render_single_mask_panel(axes[1, 0], mixed_after_tile, mixed_view, f"mixed after doublet tile ({tile_row},{tile_col})", PANEL_COLORS["mixed_after"])
    _render_single_mask_panel(axes[1, 1], mapped_mixed_tile, mixed_view, f"mapped mixed tile ({tile_row},{tile_col})", PANEL_COLORS["mapped_mixed"])
    _render_source_mask_panel(axes[1, 2], mc_source_tile, p5_source_tile, mixed_view, f"mapped source tile ({tile_row},{tile_col})")

    fig.suptitle("Spatial cell visualization", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_filtered_mixed_outputs(
    molecules: pd.DataFrame,
    output_gem: Path,
    output_parquet: Path,
    header_lines: list[str],
) -> None:
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    molecules.to_parquet(output_parquet, index=False)
    with output_gem.open("w", encoding="utf-8") as handle:
        for line in header_lines:
            handle.write(f"{line}\n")
        molecules.loc[:, GEM_COLUMNS].to_csv(handle, sep="\t", index=False)


def build_cell_gene_matrix(
    molecules: pd.DataFrame,
    cell_order: list[str],
    gene_order: list[str],
) -> sparse.csr_matrix:
    if not cell_order or not gene_order or molecules.empty:
        return sparse.csr_matrix((len(cell_order), len(gene_order)), dtype=np.int32)

    grouped = (
        molecules.groupby(["cell_label", "geneID"], as_index=False)
        .agg(mid_count=("MIDCount", "sum"))
    )
    cell_index = {cell_label: idx for idx, cell_label in enumerate(cell_order)}
    gene_index = {gene_id: idx for idx, gene_id in enumerate(gene_order)}

    grouped = grouped.loc[
        grouped["cell_label"].isin(cell_index) & grouped["geneID"].isin(gene_index)
    ].copy()
    if grouped.empty:
        return sparse.csr_matrix((len(cell_order), len(gene_order)), dtype=np.int32)

    row_idx = grouped["cell_label"].map(cell_index).to_numpy(dtype=np.int32)
    col_idx = grouped["geneID"].map(gene_index).to_numpy(dtype=np.int32)
    data = grouped["mid_count"].to_numpy(dtype=np.int32)
    return sparse.csr_matrix((data, (row_idx, col_idx)), shape=(len(cell_order), len(gene_order)))


def build_mapped_h5ad_inputs(
    mixed_cells: pd.DataFrame,
    filtered_molecules: pd.DataFrame,
    mapping_df: pd.DataFrame,
    mc_molecules: pd.DataFrame,
    p5_molecules: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Index, sparse.csr_matrix, sparse.csr_matrix]:
    kept_mapping = mapping_df.loc[
        (~mapping_df["is_doublet"])
        & mapping_df["mapped_source_dataset"].ne("")
        & mapping_df["mapped_source_label"].ne("")
    ].copy()
    if kept_mapping.empty:
        obs_columns = [
            "mixed_cell_label",
            "mapped_source_dataset",
            "mapped_source_label",
            "mapped_iou",
            "mapped_intersection_area",
            "is_doublet",
        ]
        empty_obs = pd.DataFrame(columns=obs_columns).set_index("mixed_cell_label", drop=False)
        empty_genes = pd.Index([], dtype="object")
        empty_matrix = sparse.csr_matrix((0, 0), dtype=np.int32)
        return empty_obs, empty_genes, empty_matrix, empty_matrix

    kept_labels = set(kept_mapping["mixed_cell_label"].tolist())
    mixed_order = mixed_cells.loc[
        mixed_cells["cell_label"].isin(kept_labels),
        "cell_label",
    ].tolist()
    kept_mapping["mixed_cell_label"] = pd.Categorical(
        kept_mapping["mixed_cell_label"],
        categories=mixed_order,
        ordered=True,
    )
    obs_df = kept_mapping.sort_values("mixed_cell_label").copy()
    obs_df["mixed_cell_label"] = obs_df["mixed_cell_label"].astype(str)
    obs_df = obs_df[
        [
            "mixed_cell_label",
            "mapped_source_dataset",
            "mapped_source_label",
            "mapped_iou",
            "mapped_intersection_area",
            "is_doublet",
            "doublet_reason",
        ]
    ].set_index("mixed_cell_label", drop=False)

    source_frames: list[pd.DataFrame] = []
    for dataset_name, source_frame in (("MC", mc_molecules), ("P5", p5_molecules)):
        dataset_mapping = obs_df.loc[
            obs_df["mapped_source_dataset"] == dataset_name,
            ["mixed_cell_label", "mapped_source_label"],
        ].copy()
        if dataset_mapping.empty:
            continue

        matched = source_frame.loc[:, ["cell_label", "geneID", "MIDCount"]].merge(
            dataset_mapping,
            left_on="cell_label",
            right_on="mapped_source_label",
            how="inner",
            sort=False,
        )
        if matched.empty:
            continue

        matched["cell_label"] = matched["mixed_cell_label"].astype(str)
        source_frames.append(matched[["cell_label", "geneID", "MIDCount"]].copy())
    source_molecules = (
        pd.concat(source_frames, ignore_index=True)
        if source_frames
        else pd.DataFrame(columns=["cell_label", "geneID", "MIDCount"])
    )

    mixed_gene_order = filtered_molecules.loc[
        filtered_molecules["cell_label"].isin(mixed_order), "geneID"
    ].astype(str)
    source_gene_order = source_molecules["geneID"].astype(str)
    gene_order = pd.Index(sorted(set(mixed_gene_order.tolist()) | set(source_gene_order.tolist())))

    mixed_x = build_cell_gene_matrix(filtered_molecules, mixed_order, gene_order.tolist())
    source_x = build_cell_gene_matrix(source_molecules, mixed_order, gene_order.tolist())
    return obs_df, gene_order, mixed_x, source_x


def write_mixed_mapped_h5ad(
    output_path: Path,
    obs_df: pd.DataFrame,
    gene_order: pd.Index,
    mixed_x: sparse.csr_matrix,
    source_x: sparse.csr_matrix,
) -> None:
    var_df = pd.DataFrame(index=gene_order.copy())
    adata = ad.AnnData(X=mixed_x, obs=obs_df.copy(), var=var_df)
    adata.layers["source"] = source_x
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)


def write_summary(
    output_path: Path,
    *,
    mc_rows: int,
    p5_rows: int,
    mixed_summary: dict[str, int],
    mc_cells: pd.DataFrame,
    p5_cells: pd.DataFrame,
    mixed_cells: pd.DataFrame,
    mapping_df: pd.DataFrame,
    filtered_molecules: pd.DataFrame,
    run_outputs: dict[str, dict[str, str]],
) -> None:
    summary = {
        "mc": {
            "row_count": mc_rows,
            "segmented_cell_count": int(len(mc_cells)),
            "cell_mask": run_outputs["Y40178MC"]["cell_mask"],
        },
        "p5": {
            "row_count": p5_rows,
            "segmented_cell_count": int(len(p5_cells)),
            "cell_mask": run_outputs["Y40178P5"]["cell_mask"],
        },
        "mixed": {
            "row_count": mixed_summary["mixed_row_count"],
            "total_mid_count": mixed_summary["mixed_mid_total"],
            "segmented_cell_count": int(len(mixed_cells)),
            "doublet_cell_count": int(mapping_df["is_doublet"].sum()) if not mapping_df.empty else 0,
            "kept_cell_count": int((~mapping_df["is_doublet"]).sum()) if not mapping_df.empty else 0,
            "filtered_row_count": int(len(filtered_molecules)),
            "cell_mask": run_outputs["mixed"]["cell_mask"],
        },
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate mixed-gem crosstalk with CellBin2 reruns.")
    parser.add_argument("--mc-gem", type=Path, default=DEFAULT_MC_GEM)
    parser.add_argument("--p5-gem", type=Path, default=DEFAULT_P5_GEM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cellbin2-python", type=Path, default=DEFAULT_CELLBIN2_PYTHON)
    parser.add_argument("--cellbin2-entry", type=Path, default=DEFAULT_CELLBIN2_ENTRY)
    parser.add_argument("--cellbin2-template", type=Path, default=DEFAULT_CELLBIN2_TEMPLATE)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.iou_threshold < 0:
        raise ValueError("--iou-threshold must be >= 0")
    for path_value, label in (
        (args.mc_gem, "MC GEM"),
        (args.p5_gem, "P5 GEM"),
        (args.cellbin2_python, "CellBin2 python"),
        (args.cellbin2_entry, "CellBin2 entry"),
        (args.cellbin2_template, "CellBin2 template"),
    ):
        if not path_value.exists():
            raise FileNotFoundError(f"missing {label}: {path_value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    log_step(f"using output directory: {args.output_dir}")
    ensure_output_dir(args.output_dir, args.force)

    mixed_gem_path = args.output_dir / "mixed.gem"
    if mixed_gem_path.exists():
        log_step(f"reusing mixed.gem: {mixed_gem_path}")
        mixed_summary = {
            "mixed_row_count": sum(len(chunk) for chunk in iter_gem_chunks(mixed_gem_path, args.chunk_size)),
            "mixed_mid_total": sum(int(chunk["MIDCount"].sum()) for chunk in iter_gem_chunks(mixed_gem_path, args.chunk_size)),
        }
    else:
        log_step(f"building mixed.gem: {mixed_gem_path}")
        mixed_summary = build_mixed_gem(args.mc_gem, args.p5_gem, mixed_gem_path, args.chunk_size)

    mc_gef_path = args.output_dir / "Y40178MC.gef"
    if not mc_gef_path.exists():
        log_step(f"converting GEM to GEF: {mc_gef_path}")
        mc_gef_path = convert_gem_to_gef(args.mc_gem, mc_gef_path)
    else:
        log_step(f"reusing GEF: {mc_gef_path}")
    p5_gef_path = args.output_dir / "Y40178P5.gef"
    if not p5_gef_path.exists():
        log_step(f"converting GEM to GEF: {p5_gef_path}")
        p5_gef_path = convert_gem_to_gef(args.p5_gem, p5_gef_path)
    else:
        log_step(f"reusing GEF: {p5_gef_path}")
    mixed_gef_path = args.output_dir / "mixed.gef"
    if not mixed_gef_path.exists():
        log_step(f"converting GEM to GEF: {mixed_gef_path}")
        mixed_gef_path = convert_gem_to_gef(mixed_gem_path, mixed_gef_path)
    else:
        log_step(f"reusing GEF: {mixed_gef_path}")
    mc_rows = sum(len(chunk) for chunk in iter_gem_chunks(args.mc_gem, args.chunk_size))
    p5_rows = sum(len(chunk) for chunk in iter_gem_chunks(args.p5_gem, args.chunk_size))

    configs_dir = args.output_dir / "configs"
    runs_dir = args.output_dir / "cellbin2"
    sample_defs = {
        "Y40178MC": mc_gef_path,
        "Y40178P5": p5_gef_path,
        "mixed": mixed_gef_path,
    }

    run_outputs: dict[str, dict[str, str]] = {}
    for sample_name, gem_path in sample_defs.items():
        config_path = prepare_cellbin2_config(
            sample_name=sample_name,
            gem_path=gem_path,
            template_path=args.cellbin2_template,
            config_path=configs_dir / f"{sample_name}.json",
        )
        run_output_dir = runs_dir / sample_name
        cell_mask_path = run_output_dir / f"{sample_name}_cell_mask.tif"
        if cell_mask_path.exists():
            log_step(f"reusing CellBin2 output for {sample_name}: {cell_mask_path}")
            run_outputs[sample_name] = get_run_output(sample_name, run_output_dir)
        else:
            log_step(f"running CellBin2 for {sample_name}")
            run_outputs[sample_name] = run_cellbin2(
                sample_name=sample_name,
                config_path=config_path,
                run_output_dir=run_output_dir,
                cellbin2_python=args.cellbin2_python,
                cellbin2_entry=args.cellbin2_entry,
            )

    log_step("loading CellBin2 masks")
    mc_labels = load_final_cell_labels(Path(run_outputs["Y40178MC"]["cell_mask"]))
    p5_labels = load_final_cell_labels(Path(run_outputs["Y40178P5"]["cell_mask"]))
    mixed_labels = load_final_cell_labels(Path(run_outputs["mixed"]["cell_mask"]))

    log_step("extracting segmented cell polygons")
    mc_cells = extract_cell_polygons(mc_labels, "MC")
    p5_cells = extract_cell_polygons(p5_labels, "P5")
    mixed_cells = extract_cell_polygons(mixed_labels, "mixed")

    log_step("assigning molecules to segmented cells")
    mc_molecules = assign_molecules_to_cells(args.mc_gem, mc_labels, "MC", args.chunk_size)
    p5_molecules = assign_molecules_to_cells(args.p5_gem, p5_labels, "P5", args.chunk_size)
    mixed_molecules = assign_molecules_to_cells(mixed_gem_path, mixed_labels, "mixed", args.chunk_size)

    log_step("mapping mixed cells to source cells")
    candidates_df, mapping_df = map_mixed_cells(
        mixed_cells=mixed_cells,
        mc_cells=mc_cells,
        p5_cells=p5_cells,
        iou_threshold=args.iou_threshold,
    )
    log_step("filtering doublet molecules")
    filtered_molecules = filter_mixed_doublets(mixed_molecules, mapping_df)

    log_step("writing intermediate parquet and GEM outputs")
    mc_cells.to_parquet(args.output_dir / "mc_cells.parquet", index=False)
    p5_cells.to_parquet(args.output_dir / "p5_cells.parquet", index=False)
    mixed_cells.to_parquet(args.output_dir / "mixed_cells.parquet", index=False)
    mc_molecules.to_parquet(args.output_dir / "mc_molecules.parquet", index=False)
    p5_molecules.to_parquet(args.output_dir / "p5_molecules.parquet", index=False)
    mixed_molecules.to_parquet(args.output_dir / "mixed_molecules.parquet", index=False)
    candidates_df.to_parquet(args.output_dir / "mixed_overlap_candidates.parquet", index=False)
    mapping_df.to_parquet(args.output_dir / "mixed_cell_mapping.parquet", index=False)
    mapping_df.loc[mapping_df["is_doublet"]].to_parquet(args.output_dir / "mixed_doublets.parquet", index=False)
    write_filtered_mixed_outputs(
        filtered_molecules,
        args.output_dir / "mixed_filtered.gem",
        args.output_dir / "mixed_filtered_molecules.parquet",
        read_gem_header(args.mc_gem),
    )
    obs_df, gene_order, mixed_x, source_x = build_mapped_h5ad_inputs(
        mixed_cells=mixed_cells,
        filtered_molecules=filtered_molecules,
        mapping_df=mapping_df,
        mc_molecules=mc_molecules,
        p5_molecules=p5_molecules,
    )
    log_step("writing mixed_maped.h5ad")
    write_mixed_mapped_h5ad(
        args.output_dir / "mixed_maped.h5ad",
        obs_df=obs_df,
        gene_order=gene_order,
        mixed_x=mixed_x,
        source_x=source_x,
    )
    spatial_tile_path = args.output_dir / DEFAULT_SPATIAL_TILE_FILENAME
    log_step(f"writing spatial visualization: {spatial_tile_path}")
    write_spatial_visualization(
        output_path=spatial_tile_path,
        mc_cells=mc_cells,
        p5_cells=p5_cells,
        mixed_cells=mixed_cells,
        mapping_df=mapping_df,
        mc_labels=mc_labels,
        p5_labels=p5_labels,
        mixed_labels=mixed_labels,
    )
    log_step("writing summary.json")
    write_summary(
        args.output_dir / "summary.json",
        mc_rows=mc_rows,
        p5_rows=p5_rows,
        mixed_summary=mixed_summary,
        mc_cells=mc_cells,
        p5_cells=p5_cells,
        mixed_cells=mixed_cells,
        mapping_df=mapping_df,
        filtered_molecules=filtered_molecules,
        run_outputs=run_outputs,
    )
    log_step("pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
