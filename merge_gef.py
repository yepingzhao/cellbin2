from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import cv2
import h5py
import numpy as np

from cellbin2.utils.cell_shape import f_main as rebuild_coded_cell_block

PathLike = Union[str, Path]
SENTINEL_COORDINATE = 32767
BLOCK_TILE_SIZE = 256
GENE_KEY = Tuple[bytes, bytes]


@dataclass(frozen=True)
class ExpressionEntry:
    """One per-cell expression entry with optional exon support."""

    gene_key: GENE_KEY
    count: int
    exon: Optional[int] = None


@dataclass
class LoadedGef:
    """In-memory representation of the cell-bin datasets required for merging."""

    path: Path
    sn: str
    bin_type: bytes
    omics: bytes
    resolution: int
    version: int
    geftool_ver: np.ndarray
    offset_x: int
    offset_y: int
    root_max_x: int
    root_max_y: int
    cell_dtype: np.dtype
    cell_exp_dtype: np.dtype
    gene_dtype: np.dtype
    gene_exp_dtype: np.dtype
    cell_rows: np.ndarray
    cell_border: np.ndarray
    cell_type_list: Optional[List[bytes]]
    cell_exon: Optional[np.ndarray]
    cell_exp_exon: Optional[np.ndarray]
    gene_exon: Optional[np.ndarray]
    gene_exp_exon: Optional[np.ndarray]
    cell_expressions: List[List[ExpressionEntry]]
    polygons: List[np.ndarray]
    bboxes: np.ndarray


@dataclass(frozen=True)
class KeptCell:
    """Source-backed cell payload used while rebuilding the merged file."""

    source: LoadedGef
    source_rank: int
    original_index: int
    row: np.void
    border: np.ndarray
    expressions: List[ExpressionEntry]
    cell_type_label: bytes
    cell_exon: Optional[int]


def _as_scalar(value: object) -> object:
    """Collapse 1-element numpy containers into Python scalars."""

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _as_scalar(value[()])
        if value.size == 1:
            return _as_scalar(value.reshape(-1)[0])
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_bytes(value: object) -> bytes:
    """Normalize HDF5 string-like values to raw bytes."""

    scalar = _as_scalar(value)
    if isinstance(scalar, bytes):
        return scalar.rstrip(b"\x00")
    if isinstance(scalar, np.bytes_):
        return bytes(scalar).rstrip(b"\x00")
    if isinstance(scalar, str):
        return scalar.encode("utf-8")
    raise TypeError(f"Expected bytes-compatible value, got {type(value)!r}")


def _as_str(value: object) -> str:
    """Normalize HDF5 string-like values to a Python string."""

    return _as_bytes(value).decode("utf-8", errors="replace")


def _dataset_if_present(group: h5py.Group, name: str) -> Optional[np.ndarray]:
    """Return the full dataset when present."""

    if name not in group:
        return None
    return group[name][:]


def decode_border_points(border_row: np.ndarray) -> np.ndarray:
    """Strip padding markers from a fixed-length relative border point array."""

    border = np.asarray(border_row, dtype=np.int32)
    valid_length = len(border)

    for index, point in enumerate(border):
        is_zero = point[0] == 0 and point[1] == 0
        is_sentinel = point[0] == SENTINEL_COORDINATE and point[1] == SENTINEL_COORDINATE

        if index < len(border) - 1:
            next_point = border[index + 1]
            next_is_zero = next_point[0] == 0 and next_point[1] == 0
            next_is_sentinel = (
                next_point[0] == SENTINEL_COORDINATE and next_point[1] == SENTINEL_COORDINATE
            )
            if (is_zero and next_is_zero) or (is_sentinel and next_is_sentinel):
                valid_length = index
                break
        elif is_zero or is_sentinel:
            valid_length = index

    return border[:valid_length]


def build_absolute_polygon(cell_row: np.void, border_row: np.ndarray) -> np.ndarray:
    """Convert relative border points into absolute contour coordinates."""

    relative = decode_border_points(border_row)
    if len(relative) == 0:
        return np.empty((0, 2), dtype=np.int32)

    anchor = np.array([int(cell_row["x"]), int(cell_row["y"])], dtype=np.int32)
    return relative.astype(np.int32) + anchor


def compute_bbox(polygon: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute the inclusive bounding box for a polygon."""

    if len(polygon) == 0:
        return (0, 0, -1, -1)

    return (
        int(np.min(polygon[:, 0])),
        int(np.min(polygon[:, 1])),
        int(np.max(polygon[:, 0])),
        int(np.max(polygon[:, 1])),
    )


def _bbox_intersects(a_bbox: Tuple[int, int, int, int], b_bbox: Tuple[int, int, int, int]) -> bool:
    """Return True when two inclusive bboxes intersect."""

    return not (
        a_bbox[2] < b_bbox[0]
        or b_bbox[2] < a_bbox[0]
        or a_bbox[3] < b_bbox[1]
        or b_bbox[3] < a_bbox[1]
    )


def polygon_iou(polygon_a: np.ndarray, polygon_b: np.ndarray) -> float:
    """Rasterize two polygons into a local mask and compute their IoU."""

    if len(polygon_a) < 3 or len(polygon_b) < 3:
        return 0.0

    min_x = min(int(np.min(polygon_a[:, 0])), int(np.min(polygon_b[:, 0])))
    min_y = min(int(np.min(polygon_a[:, 1])), int(np.min(polygon_b[:, 1])))
    max_x = max(int(np.max(polygon_a[:, 0])), int(np.max(polygon_b[:, 0])))
    max_y = max(int(np.max(polygon_a[:, 1])), int(np.max(polygon_b[:, 1])))

    width = max_x - min_x + 3
    height = max_y - min_y + 3
    shift = np.array([1 - min_x, 1 - min_y], dtype=np.int32)

    mask_a = np.zeros((height, width), dtype=np.uint8)
    mask_b = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask_a, [(polygon_a + shift).astype(np.int32)], 1)
    cv2.fillPoly(mask_b, [(polygon_b + shift).astype(np.int32)], 1)

    intersection = np.count_nonzero(mask_a & mask_b)
    if intersection == 0:
        return 0.0

    union = np.count_nonzero(mask_a | mask_b)
    if union == 0:
        return 0.0

    return float(intersection / union)


def _block_id(x: int, y: int, x_blocks: int) -> int:
    """Flatten a cell position to the blockIndex ordering used by cellBin files."""

    return (y // BLOCK_TILE_SIZE) * x_blocks + (x // BLOCK_TILE_SIZE)


def _tiles_for_bbox(bbox: Tuple[int, int, int, int]) -> Iterable[Tuple[int, int]]:
    """Yield all 256x256 tiles touched by the bbox."""

    min_x, min_y, max_x, max_y = bbox
    if max_x < min_x or max_y < min_y:
        return []

    x_start = min_x // BLOCK_TILE_SIZE
    x_stop = max_x // BLOCK_TILE_SIZE
    y_start = min_y // BLOCK_TILE_SIZE
    y_stop = max_y // BLOCK_TILE_SIZE

    return (
        (tile_x, tile_y)
        for tile_x in range(x_start, x_stop + 1)
        for tile_y in range(y_start, y_stop + 1)
    )


def _resolve_cell_type_label(cell_type_list: Optional[List[bytes]], cell_type_id: int) -> bytes:
    """Resolve a cellTypeID to a stable label for cross-file remapping."""

    if cell_type_list is not None and 0 <= cell_type_id < len(cell_type_list):
        return cell_type_list[cell_type_id]
    return f"cellTypeID:{cell_type_id}".encode("utf-8")


def _load_cell_type_list(cell_bin: h5py.Group) -> Optional[List[bytes]]:
    """Read the optional cellTypeList dataset."""

    if "cellTypeList" not in cell_bin:
        return None
    return [_as_bytes(item) for item in cell_bin["cellTypeList"][:]]


def _build_cell_expressions(
    cell_rows: np.ndarray,
    cell_exp: np.ndarray,
    gene_rows: np.ndarray,
    cell_exp_exon: Optional[np.ndarray],
) -> List[List[ExpressionEntry]]:
    """Expand per-cell expression slices into Python lists keyed by gene identity."""

    expressions: List[List[ExpressionEntry]] = []

    for cell in cell_rows:
        start = int(cell["offset"])
        stop = start + int(cell["geneCount"])
        slice_entries = cell_exp[start:stop]
        exon_slice = cell_exp_exon[start:stop] if cell_exp_exon is not None else None
        expanded: List[ExpressionEntry] = []

        for idx, entry in enumerate(slice_entries):
            gene_index = int(entry["geneID"])
            gene_row = gene_rows[gene_index]
            gene_key = (_as_bytes(gene_row["geneID"]), _as_bytes(gene_row["geneName"]))
            expanded.append(
                ExpressionEntry(
                    gene_key=gene_key,
                    count=int(entry["count"]),
                    exon=int(exon_slice[idx]) if exon_slice is not None else None,
                )
            )

        expressions.append(expanded)

    return expressions


def _load_polygons(cell_rows: np.ndarray, cell_border: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Precompute absolute polygons and bboxes for all cells."""

    polygons: List[np.ndarray] = []
    bbox_rows: List[Tuple[int, int, int, int]] = []

    for cell_row, border_row in zip(cell_rows, cell_border):
        polygon = build_absolute_polygon(cell_row, border_row)
        polygons.append(polygon)
        bbox_rows.append(compute_bbox(polygon))

    return polygons, np.asarray(bbox_rows, dtype=np.int32)


def load_cellbin_gef(path: PathLike) -> LoadedGef:
    """Load the cell-bin datasets needed to filter and rebuild a merged file."""

    resolved = Path(path)
    with h5py.File(resolved, "r") as handle:
        cell_bin = handle["cellBin"]
        cell_rows = cell_bin["cell"][:]
        gene_rows = cell_bin["gene"][:]
        cell_exp = cell_bin["cellExp"][:]
        cell_border = cell_bin["cellBorder"][:]
        cell_exp_exon = _dataset_if_present(cell_bin, "cellExpExon")
        cell_exon = _dataset_if_present(cell_bin, "cellExon")
        gene_exon = _dataset_if_present(cell_bin, "geneExon")
        gene_exp_exon = _dataset_if_present(cell_bin, "geneExpExon")

        cell_expressions = _build_cell_expressions(cell_rows, cell_exp, gene_rows, cell_exp_exon)
        polygons, bboxes = _load_polygons(cell_rows, cell_border)

        return LoadedGef(
            path=resolved,
            sn=_as_str(handle.attrs["sn"]),
            bin_type=_as_bytes(handle.attrs["bin_type"]),
            omics=_as_bytes(handle.attrs["omics"]),
            resolution=int(_as_scalar(handle.attrs["resolution"])),
            version=int(_as_scalar(handle.attrs["version"])),
            geftool_ver=np.asarray(handle.attrs["geftool_ver"]),
            offset_x=int(_as_scalar(handle.attrs["offsetX"])),
            offset_y=int(_as_scalar(handle.attrs["offsetY"])),
            root_max_x=int(_as_scalar(handle.attrs["maxX"])),
            root_max_y=int(_as_scalar(handle.attrs["maxY"])),
            cell_dtype=cell_rows.dtype,
            cell_exp_dtype=cell_exp.dtype,
            gene_dtype=gene_rows.dtype,
            gene_exp_dtype=cell_bin["geneExp"].dtype,
            cell_rows=cell_rows,
            cell_border=cell_border,
            cell_type_list=_load_cell_type_list(cell_bin),
            cell_exon=cell_exon,
            cell_exp_exon=cell_exp_exon,
            gene_exon=gene_exon,
            gene_exp_exon=gene_exp_exon,
            cell_expressions=cell_expressions,
            polygons=polygons,
            bboxes=bboxes,
        )


def _validate_merge_compatibility(data_a: LoadedGef, data_b: LoadedGef) -> None:
    """Raise when the two input files are incompatible for direct merging."""

    if data_a.bin_type != b"CellBin" or data_b.bin_type != b"CellBin":
        raise ValueError("Both inputs must be cell-bin GEF files.")
    if data_a.omics != data_b.omics:
        raise ValueError("Both inputs must have the same omics type.")
    if data_a.resolution != data_b.resolution:
        raise ValueError("Both inputs must have the same resolution.")
    if data_a.version != data_b.version:
        raise ValueError("Both inputs must have the same GEF version.")
    if data_a.cell_dtype != data_b.cell_dtype:
        raise ValueError("cell dataset schema mismatch between input files.")
    if data_a.gene_dtype != data_b.gene_dtype:
        raise ValueError("gene dataset schema mismatch between input files.")
    if data_a.cell_exp_dtype != data_b.cell_exp_dtype:
        raise ValueError("cellExp dataset schema mismatch between input files.")
    if data_a.gene_exp_dtype != data_b.gene_exp_dtype:
        raise ValueError("geneExp dataset schema mismatch between input files.")
    if data_a.cell_border.shape[1:] != data_b.cell_border.shape[1:]:
        raise ValueError("cellBorder dataset shape mismatch between input files.")

    optional_datasets = (
        ("cellExon", data_a.cell_exon, data_b.cell_exon),
        ("cellExpExon", data_a.cell_exp_exon, data_b.cell_exp_exon),
        ("geneExon", data_a.gene_exon, data_b.gene_exon),
        ("geneExpExon", data_a.gene_exp_exon, data_b.gene_exp_exon),
    )
    for name, first, second in optional_datasets:
        if (first is None) != (second is None):
            raise ValueError(f"Optional dataset mismatch: {name} must exist in both files or neither.")


def find_overlapping_pairs(
    data_a: LoadedGef,
    data_b: LoadedGef,
    iou_threshold: float,
) -> Tuple[Set[int], Set[int], int]:
    """Find cross-file duplicates using contour overlap only."""

    tile_index: Dict[Tuple[int, int], List[int]] = {}
    for cell_index, bbox_row in enumerate(data_b.bboxes):
        bbox = tuple(int(value) for value in bbox_row)
        for tile in _tiles_for_bbox(bbox):
            tile_index.setdefault(tile, []).append(cell_index)

    remove_a: Set[int] = set()
    remove_b: Set[int] = set()
    pair_count = 0

    for index_a, bbox_row_a in enumerate(data_a.bboxes):
        bbox_a = tuple(int(value) for value in bbox_row_a)
        candidates: Set[int] = set()
        for tile in _tiles_for_bbox(bbox_a):
            candidates.update(tile_index.get(tile, ()))

        for index_b in sorted(candidates):
            bbox_b = tuple(int(value) for value in data_b.bboxes[index_b])
            if not _bbox_intersects(bbox_a, bbox_b):
                continue

            overlap = polygon_iou(data_a.polygons[index_a], data_b.polygons[index_b])
            if overlap >= iou_threshold:
                if index_a not in remove_a or index_b not in remove_b:
                    pair_count += 1
                remove_a.add(index_a)
                remove_b.add(index_b)

    return remove_a, remove_b, pair_count


def _collect_kept_cells(
    data_a: LoadedGef,
    data_b: LoadedGef,
    remove_a: Set[int],
    remove_b: Set[int],
) -> List[KeptCell]:
    """Build source-backed cell objects for all retained cells."""

    kept_cells: List[KeptCell] = []

    for source_rank, (loaded, removed) in enumerate(((data_a, remove_a), (data_b, remove_b))):
        for index, row in enumerate(loaded.cell_rows):
            if index in removed:
                continue
            kept_cells.append(
                KeptCell(
                    source=loaded,
                    source_rank=source_rank,
                    original_index=index,
                    row=row,
                    border=loaded.cell_border[index],
                    expressions=loaded.cell_expressions[index],
                    cell_type_label=_resolve_cell_type_label(
                        loaded.cell_type_list,
                        int(row["cellTypeID"]),
                    ),
                    cell_exon=int(loaded.cell_exon[index]) if loaded.cell_exon is not None else None,
                )
            )

    return kept_cells


def _set_cell_dataset_attrs(dataset: h5py.Dataset, cell_rows: np.ndarray) -> None:
    """Populate summary attrs for the rebuilt cell dataset."""

    if len(cell_rows) == 0:
        dataset.attrs["averageArea"] = np.array([0.0], dtype=np.float32)
        dataset.attrs["averageDnbCount"] = np.array([0.0], dtype=np.float32)
        dataset.attrs["averageExpCount"] = np.array([0.0], dtype=np.float32)
        dataset.attrs["averageGeneCount"] = np.array([0.0], dtype=np.float32)
        dataset.attrs["maxArea"] = np.array([0], dtype=np.uint16)
        dataset.attrs["maxDnbCount"] = np.array([0], dtype=np.uint16)
        dataset.attrs["maxExpCount"] = np.array([0], dtype=np.uint16)
        dataset.attrs["maxGeneCount"] = np.array([0], dtype=np.uint16)
        dataset.attrs["maxX"] = np.array([0], dtype=np.int32)
        dataset.attrs["maxY"] = np.array([0], dtype=np.int32)
        dataset.attrs["medianArea"] = np.array([0.0], dtype=np.float32)
        dataset.attrs["medianDnbCount"] = np.array([0.0], dtype=np.float32)
        dataset.attrs["medianExpCount"] = np.array([0.0], dtype=np.float32)
        dataset.attrs["medianGeneCount"] = np.array([0.0], dtype=np.float32)
        dataset.attrs["minArea"] = np.array([0], dtype=np.uint16)
        dataset.attrs["minDnbCount"] = np.array([0], dtype=np.uint16)
        dataset.attrs["minExpCount"] = np.array([0], dtype=np.uint16)
        dataset.attrs["minGeneCount"] = np.array([0], dtype=np.uint16)
        dataset.attrs["minX"] = np.array([0], dtype=np.int32)
        dataset.attrs["minY"] = np.array([0], dtype=np.int32)
        return

    dataset.attrs["averageArea"] = np.array([float(np.mean(cell_rows["area"]))], dtype=np.float32)
    dataset.attrs["averageDnbCount"] = np.array([float(np.mean(cell_rows["dnbCount"]))], dtype=np.float32)
    dataset.attrs["averageExpCount"] = np.array([float(np.mean(cell_rows["expCount"]))], dtype=np.float32)
    dataset.attrs["averageGeneCount"] = np.array([float(np.mean(cell_rows["geneCount"]))], dtype=np.float32)
    dataset.attrs["maxArea"] = np.array([int(np.max(cell_rows["area"]))], dtype=np.uint16)
    dataset.attrs["maxDnbCount"] = np.array([int(np.max(cell_rows["dnbCount"]))], dtype=np.uint16)
    dataset.attrs["maxExpCount"] = np.array([int(np.max(cell_rows["expCount"]))], dtype=np.uint16)
    dataset.attrs["maxGeneCount"] = np.array([int(np.max(cell_rows["geneCount"]))], dtype=np.uint16)
    dataset.attrs["maxX"] = np.array([int(np.max(cell_rows["x"]))], dtype=np.int32)
    dataset.attrs["maxY"] = np.array([int(np.max(cell_rows["y"]))], dtype=np.int32)
    dataset.attrs["medianArea"] = np.array([float(np.median(cell_rows["area"]))], dtype=np.float32)
    dataset.attrs["medianDnbCount"] = np.array([float(np.median(cell_rows["dnbCount"]))], dtype=np.float32)
    dataset.attrs["medianExpCount"] = np.array([float(np.median(cell_rows["expCount"]))], dtype=np.float32)
    dataset.attrs["medianGeneCount"] = np.array([float(np.median(cell_rows["geneCount"]))], dtype=np.float32)
    dataset.attrs["minArea"] = np.array([int(np.min(cell_rows["area"]))], dtype=np.uint16)
    dataset.attrs["minDnbCount"] = np.array([int(np.min(cell_rows["dnbCount"]))], dtype=np.uint16)
    dataset.attrs["minExpCount"] = np.array([int(np.min(cell_rows["expCount"]))], dtype=np.uint16)
    dataset.attrs["minGeneCount"] = np.array([int(np.min(cell_rows["geneCount"]))], dtype=np.uint16)
    dataset.attrs["minX"] = np.array([int(np.min(cell_rows["x"]))], dtype=np.int32)
    dataset.attrs["minY"] = np.array([int(np.min(cell_rows["y"]))], dtype=np.int32)


def _set_gene_dataset_attrs(dataset: h5py.Dataset, gene_rows: np.ndarray) -> None:
    """Populate summary attrs for the rebuilt gene dataset."""

    if len(gene_rows) == 0:
        dataset.attrs["maxCellCount"] = np.array([0], dtype=np.uint32)
        dataset.attrs["maxExpCount"] = np.array([0], dtype=np.uint32)
        dataset.attrs["minCellCount"] = np.array([0], dtype=np.uint32)
        dataset.attrs["minExpCount"] = np.array([0], dtype=np.uint32)
        return

    dataset.attrs["maxCellCount"] = np.array([int(np.max(gene_rows["cellCount"]))], dtype=np.uint32)
    dataset.attrs["maxExpCount"] = np.array([int(np.max(gene_rows["expCount"]))], dtype=np.uint32)
    dataset.attrs["minCellCount"] = np.array([int(np.min(gene_rows["cellCount"]))], dtype=np.uint32)
    dataset.attrs["minExpCount"] = np.array([int(np.min(gene_rows["expCount"]))], dtype=np.uint32)


def _write_empty_coded_cell_block(output: Path, resolution: int) -> None:
    """Create a minimal empty codedCellBlock when no cells survive filtering."""

    info = {
        "@type": "neuroglancer_annotations_v1",
        "annotation_type": "CELL_SHAPE",
        "dimensions": {"x": [resolution, "nm"], "y": [resolution, "nm"]},
        "by_id": {"key": "by_id"},
        "relationships": [],
        "lower_bound": [0.0, 0.0],
        "upper_bound": [0.0, 0.0],
        "properties": [
            {"id": "id", "type": "uint32"},
            {"id": "geneCount", "type": "uint32"},
            {"id": "expCount", "type": "uint32"},
            {"id": "dnbCount", "type": "uint32"},
            {"id": "area", "type": "uint32"},
            {"id": "cellTypeID", "type": "uint32"},
            {"id": "clusterID", "type": "uint32"},
        ],
        "spatial": [{"chunk_size": [256.0, 256.0], "grid_shape": [1, 1], "key": "L0", "limit": 10000}],
        "emptyChunk": {"0": ["0,0"]},
    }

    with h5py.File(output, "a") as handle:
        coded = handle.create_group("codedCellBlock")
        coded.create_group("L0")
        coded.attrs["info"] = json.dumps(info)


def finalize_coded_cell_block(output: Path, cell_count: int, resolution: int) -> None:
    """Generate the codedCellBlock view for downstream tooling."""

    if cell_count == 0:
        _write_empty_coded_cell_block(output, resolution)
        return

    rebuild_coded_cell_block(str(output))


def rebuild_filtered_gef(
    data_a: LoadedGef,
    data_b: LoadedGef,
    remove_a: Set[int],
    remove_b: Set[int],
    output: PathLike,
) -> Dict[str, int]:
    """Rebuild a valid merged cell-bin GEF from the retained cells."""

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)

    kept_cells = _collect_kept_cells(data_a, data_b, remove_a, remove_b)
    root_max_x = max(data_a.root_max_x, data_b.root_max_x)
    root_max_y = max(data_a.root_max_y, data_b.root_max_y)
    x_blocks = root_max_x // BLOCK_TILE_SIZE + 1
    y_blocks = root_max_y // BLOCK_TILE_SIZE + 1

    kept_cells.sort(
        key=lambda cell: (
            _block_id(int(cell.row["x"]), int(cell.row["y"]), x_blocks),
            cell.source_rank,
            cell.original_index,
        )
    )

    merged_cell_type_labels: Dict[bytes, int] = {}
    merged_gene_indices: Dict[GENE_KEY, int] = {}
    gene_assignments: Dict[GENE_KEY, List[Tuple[int, int, Optional[int]]]] = {}

    cell_rows_out = np.zeros(len(kept_cells), dtype=data_a.cell_dtype)
    cell_border_out = np.full((len(kept_cells), 32, 2), SENTINEL_COORDINATE, dtype=np.int16)
    cell_exp_rows: List[Tuple[int, int]] = []
    cell_exp_exon_rows: List[int] = []
    cell_exon_rows: List[int] = []

    for new_cell_id, kept_cell in enumerate(kept_cells):
        merged_cell_type_id = merged_cell_type_labels.setdefault(
            kept_cell.cell_type_label,
            len(merged_cell_type_labels),
        )
        cell_exp_offset = len(cell_exp_rows)
        expression_count = sum(entry.count for entry in kept_cell.expressions)
        gene_count = len(kept_cell.expressions)

        cell_rows_out[new_cell_id] = (
            new_cell_id,
            int(kept_cell.row["x"]),
            int(kept_cell.row["y"]),
            cell_exp_offset,
            gene_count,
            expression_count,
            int(kept_cell.row["dnbCount"]),
            int(kept_cell.row["area"]),
            merged_cell_type_id,
            int(kept_cell.row["clusterID"]),
        )
        cell_border_out[new_cell_id] = kept_cell.border

        if kept_cell.cell_exon is not None:
            cell_exon_rows.append(kept_cell.cell_exon)

        for entry in kept_cell.expressions:
            gene_index = merged_gene_indices.setdefault(entry.gene_key, len(merged_gene_indices))
            cell_exp_rows.append((gene_index, entry.count))
            if data_a.cell_exp_exon is not None:
                cell_exp_exon_rows.append(0 if entry.exon is None else int(entry.exon))
            gene_assignments.setdefault(entry.gene_key, []).append((new_cell_id, entry.count, entry.exon))

    gene_rows_out = np.zeros(len(merged_gene_indices), dtype=data_a.gene_dtype)
    gene_exp_rows: List[Tuple[int, int]] = []
    gene_exp_exon_rows: List[int] = []
    gene_exon_rows = np.zeros(len(merged_gene_indices), dtype=np.uint32) if data_a.gene_exon is not None else None

    for gene_key, gene_index in sorted(merged_gene_indices.items(), key=lambda item: item[1]):
        assignments = gene_assignments.get(gene_key, [])
        gene_exp_offset = len(gene_exp_rows)
        counts = [count for _, count, _ in assignments]
        gene_rows_out[gene_index] = (
            gene_key[0],
            gene_key[1],
            gene_exp_offset,
            len(assignments),
            int(sum(counts)),
            max(counts) if counts else 0,
        )
        for cell_id, count, exon in assignments:
            gene_exp_rows.append((cell_id, count))
            if data_a.gene_exp_exon is not None:
                gene_exp_exon_rows.append(0 if exon is None else int(exon))
        if gene_exon_rows is not None:
            gene_exon_rows[gene_index] = int(sum(0 if exon is None else exon for _, _, exon in assignments))

    cell_exp_out = np.asarray(cell_exp_rows, dtype=data_a.cell_exp_dtype)
    gene_exp_out = np.asarray(gene_exp_rows, dtype=data_a.gene_exp_dtype)
    block_counts = np.zeros(x_blocks * y_blocks, dtype=np.uint32)
    for cell_row in cell_rows_out:
        block_counts[_block_id(int(cell_row["x"]), int(cell_row["y"]), x_blocks)] += 1
    block_index = np.zeros(len(block_counts) + 1, dtype=np.uint32)
    block_index[1:] = np.cumsum(block_counts, dtype=np.uint32)

    if target.exists():
        target.unlink()

    with h5py.File(target, "w") as handle:
        handle.attrs["bin_type"] = np.array([b"CellBin"], dtype="S32")
        handle.attrs["geftool_ver"] = np.asarray(data_a.geftool_ver)
        handle.attrs["maxX"] = np.array([root_max_x], dtype=np.int32)
        handle.attrs["maxY"] = np.array([root_max_y], dtype=np.int32)
        handle.attrs["offsetX"] = np.array([data_a.offset_x], dtype=np.int32)
        handle.attrs["offsetY"] = np.array([data_a.offset_y], dtype=np.int32)
        handle.attrs["omics"] = np.array([data_a.omics], dtype="S32")
        handle.attrs["resolution"] = np.array([data_a.resolution], dtype=np.uint32)
        handle.attrs["sn"] = f"{data_a.sn}__{data_b.sn}_merged"
        handle.attrs["version"] = np.array([data_a.version], dtype=np.uint32)

        cell_bin = handle.create_group("cellBin")
        cell_bin.create_dataset(
            "blockSize",
            data=np.array([BLOCK_TILE_SIZE, BLOCK_TILE_SIZE, x_blocks, y_blocks], dtype=np.uint32),
        )
        cell_bin.create_dataset("blockIndex", data=block_index)

        cell_dataset = cell_bin.create_dataset("cell", data=cell_rows_out)
        _set_cell_dataset_attrs(cell_dataset, cell_rows_out)
        cell_bin.create_dataset("cellBorder", data=cell_border_out)

        cell_exp_dataset = cell_bin.create_dataset("cellExp", data=cell_exp_out)
        max_cell_exp = int(np.max(cell_exp_out["count"])) if len(cell_exp_out) > 0 else 0
        cell_exp_dataset.attrs["maxCount"] = np.array([max_cell_exp], dtype=np.uint16)

        gene_dataset = cell_bin.create_dataset("gene", data=gene_rows_out)
        _set_gene_dataset_attrs(gene_dataset, gene_rows_out)

        gene_exp_dataset = cell_bin.create_dataset("geneExp", data=gene_exp_out)
        max_gene_exp = int(np.max(gene_exp_out["count"])) if len(gene_exp_out) > 0 else 0
        gene_exp_dataset.attrs["maxCount"] = np.array([max_gene_exp], dtype=np.uint16)

        if data_a.cell_exon is not None:
            cell_bin.create_dataset("cellExon", data=np.asarray(cell_exon_rows, dtype=data_a.cell_exon.dtype))
        if data_a.cell_exp_exon is not None:
            cell_bin.create_dataset(
                "cellExpExon",
                data=np.asarray(cell_exp_exon_rows, dtype=data_a.cell_exp_exon.dtype),
            )
        if gene_exon_rows is not None:
            cell_bin.create_dataset("geneExon", data=np.asarray(gene_exon_rows, dtype=data_a.gene_exon.dtype))
        if data_a.gene_exp_exon is not None:
            cell_bin.create_dataset(
                "geneExpExon",
                data=np.asarray(gene_exp_exon_rows, dtype=data_a.gene_exp_exon.dtype),
            )

        max_label_width = max((len(label) for label in merged_cell_type_labels), default=1)
        cell_type_dtype = f"S{max(32, max_label_width)}"
        ordered_labels = [
            label for label, _ in sorted(merged_cell_type_labels.items(), key=lambda item: item[1])
        ]
        cell_bin.create_dataset("cellTypeList", data=np.asarray(ordered_labels, dtype=cell_type_dtype))

    finalize_coded_cell_block(target, len(kept_cells), data_a.resolution)

    return {
        "input_cells_a": int(len(data_a.cell_rows)),
        "input_cells_b": int(len(data_b.cell_rows)),
        "removed_cells_a": int(len(remove_a)),
        "removed_cells_b": int(len(remove_b)),
        "kept_cells": int(len(kept_cells)),
    }


def merge_gefs(
    input_a: PathLike,
    input_b: PathLike,
    output: PathLike,
    iou_threshold: float = 0.3,
) -> Dict[str, object]:
    """Merge two cell-bin GEF files after removing overlapping cross-file cells."""

    data_a = load_cellbin_gef(input_a)
    data_b = load_cellbin_gef(input_b)
    _validate_merge_compatibility(data_a, data_b)

    remove_a, remove_b, pair_count = find_overlapping_pairs(data_a, data_b, iou_threshold)
    summary = rebuild_filtered_gef(data_a, data_b, remove_a, remove_b, output)
    summary["overlapping_pairs"] = pair_count
    summary["iou_threshold"] = float(iou_threshold)
    summary["output_path"] = str(Path(output))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the merge command."""

    parser = argparse.ArgumentParser(
        description="Merge two cell-bin GEF files by removing cross-file contour overlaps.",
    )
    parser.add_argument("--input-a", type=Path, required=True, help="First input .cellbin.gef file.")
    parser.add_argument("--input-b", type=Path, required=True, help="Second input .cellbin.gef file.")
    parser.add_argument("--output", type=Path, required=True, help="Output merged .cellbin.gef file.")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Contour IoU threshold used to remove overlapping cells.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = merge_gefs(
            args.input_a,
            args.input_b,
            args.output,
            iou_threshold=args.iou_threshold,
        )
    except Exception as exc:
        print(f"Failed to merge GEF files: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
