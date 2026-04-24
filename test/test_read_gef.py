import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CELL_DTYPE = np.dtype(
    [
        ("id", "<u4"),
        ("x", "<i4"),
        ("y", "<i4"),
        ("offset", "<u4"),
        ("geneCount", "<u2"),
        ("expCount", "<u2"),
        ("dnbCount", "<u2"),
        ("area", "<u2"),
        ("cellTypeID", "<u2"),
        ("clusterID", "<u2"),
    ]
)
CELL_EXP_DTYPE = np.dtype([("geneID", "<u4"), ("count", "<u2")])
GENE_DTYPE = np.dtype(
    [
        ("geneID", "S64"),
        ("geneName", "S64"),
        ("offset", "<u4"),
        ("cellCount", "<u4"),
        ("expCount", "<u4"),
        ("maxMIDcount", "<u2"),
    ]
)
GENE_EXP_DTYPE = np.dtype([("cellID", "<u4"), ("count", "<u2")])
SENTINEL = 32767


def try_load_module():
    spec = importlib.util.spec_from_file_location("read_gef", ROOT / "read_gef.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - exercised in red phase
        sys.modules.pop(spec.name, None)
        return None, exc

    return module, None


def _polygon_area(points: Iterable[Tuple[int, int]]) -> int:
    polygon = np.asarray(list(points), dtype=np.int64)
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    twice_area = np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))
    return max(1, int(abs(twice_area) // 2))


def _relative_border(points: List[Tuple[int, int]]) -> np.ndarray:
    border = np.full((32, 2), SENTINEL, dtype=np.int16)
    border[: len(points)] = np.asarray(points, dtype=np.int16)
    return border


def _write_mini_cellbin_gef(path: Path, sn: str, cells: List[Dict]) -> None:
    gene_order: Dict[str, int] = {}
    for cell in cells:
        for gene_name in cell["expr"]:
            gene_order.setdefault(gene_name, len(gene_order))

    cell_rows = np.zeros(len(cells), dtype=CELL_DTYPE)
    cell_borders = np.full((len(cells), 32, 2), SENTINEL, dtype=np.int16)
    cell_exp_rows: List[Tuple[int, int]] = []
    gene_to_cells: Dict[int, List[Tuple[int, int]]] = {gene_idx: [] for gene_idx in gene_order.values()}

    min_x = min(cell["x"] for cell in cells)
    min_y = min(cell["y"] for cell in cells)
    max_x = max(cell["x"] for cell in cells)
    max_y = max(cell["y"] for cell in cells)

    for cell_index, cell in enumerate(cells):
        expr_items = list(cell["expr"].items())
        offset = len(cell_exp_rows)
        exp_count = sum(count for _, count in expr_items)
        area = _polygon_area(cell["border"])
        cell_rows[cell_index] = (
            cell_index,
            cell["x"],
            cell["y"],
            offset,
            len(expr_items),
            exp_count,
            cell.get("dnbCount", exp_count),
            area,
            cell.get("cellTypeID", 0),
            cell.get("clusterID", 0),
        )
        cell_borders[cell_index] = _relative_border(cell["border"])

        for gene_name, count in expr_items:
            gene_index = gene_order[gene_name]
            cell_exp_rows.append((gene_index, count))
            gene_to_cells[gene_index].append((cell_index, count))

    cell_exp = np.asarray(cell_exp_rows, dtype=CELL_EXP_DTYPE)
    gene_rows = np.zeros(len(gene_order), dtype=GENE_DTYPE)
    gene_exp_rows: List[Tuple[int, int]] = []

    sorted_genes = sorted(gene_order.items(), key=lambda item: item[1])
    for gene_name, gene_index in sorted_genes:
        assignments = gene_to_cells[gene_index]
        offset = len(gene_exp_rows)
        max_mid = max(count for _, count in assignments)
        exp_count = sum(count for _, count in assignments)
        gene_rows[gene_index] = (
            str(gene_index).encode("utf-8"),
            gene_name.encode("utf-8"),
            offset,
            len(assignments),
            exp_count,
            max_mid,
        )
        gene_exp_rows.extend(assignments)

    gene_exp = np.asarray(gene_exp_rows, dtype=GENE_EXP_DTYPE)

    with h5py.File(path, "w") as handle:
        handle.attrs["bin_type"] = np.array([b"CellBin"], dtype="S32")
        handle.attrs["geftool_ver"] = np.array([1, 2, 5], dtype=np.uint32)
        handle.attrs["maxX"] = np.array([max_x], dtype=np.int32)
        handle.attrs["maxY"] = np.array([max_y], dtype=np.int32)
        handle.attrs["offsetX"] = np.array([0], dtype=np.int32)
        handle.attrs["offsetY"] = np.array([0], dtype=np.int32)
        handle.attrs["omics"] = np.array([b"Transcriptomics"], dtype="S32")
        handle.attrs["resolution"] = np.array([500], dtype=np.uint32)
        handle.attrs["sn"] = sn
        handle.attrs["version"] = np.array([4], dtype=np.uint32)

        cell_bin = handle.create_group("cellBin")
        cell_bin.create_dataset("blockSize", data=np.array([256, 256, 1, 1], dtype=np.uint32))
        cell_bin.create_dataset("blockIndex", data=np.array([0, len(cells)], dtype=np.uint32))
        cell_dataset = cell_bin.create_dataset("cell", data=cell_rows)
        cell_dataset.attrs["averageArea"] = np.array([float(np.mean(cell_rows["area"]))], dtype=np.float32)
        cell_dataset.attrs["averageDnbCount"] = np.array([float(np.mean(cell_rows["dnbCount"]))], dtype=np.float32)
        cell_dataset.attrs["averageExpCount"] = np.array([float(np.mean(cell_rows["expCount"]))], dtype=np.float32)
        cell_dataset.attrs["averageGeneCount"] = np.array([float(np.mean(cell_rows["geneCount"]))], dtype=np.float32)
        cell_dataset.attrs["maxArea"] = np.array([int(np.max(cell_rows["area"]))], dtype=np.uint16)
        cell_dataset.attrs["maxDnbCount"] = np.array([int(np.max(cell_rows["dnbCount"]))], dtype=np.uint16)
        cell_dataset.attrs["maxExpCount"] = np.array([int(np.max(cell_rows["expCount"]))], dtype=np.uint16)
        cell_dataset.attrs["maxGeneCount"] = np.array([int(np.max(cell_rows["geneCount"]))], dtype=np.uint16)
        cell_dataset.attrs["maxX"] = np.array([max_x], dtype=np.int32)
        cell_dataset.attrs["maxY"] = np.array([max_y], dtype=np.int32)
        cell_dataset.attrs["medianArea"] = np.array([float(np.median(cell_rows["area"]))], dtype=np.float32)
        cell_dataset.attrs["medianDnbCount"] = np.array([float(np.median(cell_rows["dnbCount"]))], dtype=np.float32)
        cell_dataset.attrs["medianExpCount"] = np.array([float(np.median(cell_rows["expCount"]))], dtype=np.float32)
        cell_dataset.attrs["medianGeneCount"] = np.array([float(np.median(cell_rows["geneCount"]))], dtype=np.float32)
        cell_dataset.attrs["minArea"] = np.array([int(np.min(cell_rows["area"]))], dtype=np.uint16)
        cell_dataset.attrs["minDnbCount"] = np.array([int(np.min(cell_rows["dnbCount"]))], dtype=np.uint16)
        cell_dataset.attrs["minExpCount"] = np.array([int(np.min(cell_rows["expCount"]))], dtype=np.uint16)
        cell_dataset.attrs["minGeneCount"] = np.array([int(np.min(cell_rows["geneCount"]))], dtype=np.uint16)
        cell_dataset.attrs["minX"] = np.array([min_x], dtype=np.int32)
        cell_dataset.attrs["minY"] = np.array([min_y], dtype=np.int32)

        cell_bin.create_dataset("cellBorder", data=cell_borders)
        cell_exp_dataset = cell_bin.create_dataset("cellExp", data=cell_exp)
        if len(cell_exp) > 0:
            cell_exp_dataset.attrs["maxCount"] = np.array([int(np.max(cell_exp["count"]))], dtype=np.uint16)
        else:
            cell_exp_dataset.attrs["maxCount"] = np.array([0], dtype=np.uint16)

        gene_dataset = cell_bin.create_dataset("gene", data=gene_rows)
        if len(gene_rows) > 0:
            gene_dataset.attrs["maxCellCount"] = np.array([int(np.max(gene_rows["cellCount"]))], dtype=np.uint32)
            gene_dataset.attrs["maxExpCount"] = np.array([int(np.max(gene_rows["expCount"]))], dtype=np.uint32)
            gene_dataset.attrs["minCellCount"] = np.array([int(np.min(gene_rows["cellCount"]))], dtype=np.uint32)
            gene_dataset.attrs["minExpCount"] = np.array([int(np.min(gene_rows["expCount"]))], dtype=np.uint32)
        else:
            gene_dataset.attrs["maxCellCount"] = np.array([0], dtype=np.uint32)
            gene_dataset.attrs["maxExpCount"] = np.array([0], dtype=np.uint32)
            gene_dataset.attrs["minCellCount"] = np.array([0], dtype=np.uint32)
            gene_dataset.attrs["minExpCount"] = np.array([0], dtype=np.uint32)

        gene_exp_dataset = cell_bin.create_dataset("geneExp", data=gene_exp)
        if len(gene_exp) > 0:
            gene_exp_dataset.attrs["maxCount"] = np.array([int(np.max(gene_exp["count"]))], dtype=np.uint16)
        else:
            gene_exp_dataset.attrs["maxCount"] = np.array([0], dtype=np.uint16)


class ReadGefMergeTest(unittest.TestCase):
    def test_module_imports_without_stereo_dependency_and_exposes_merge_api(self):
        module, error = try_load_module()

        self.assertIsNone(error, f"failed to import read_gef.py: {error}")
        self.assertTrue(hasattr(module, "merge_gefs"))

    def test_decode_border_points_strips_sentinel_padding(self):
        module, error = try_load_module()
        self.assertIsNone(error, f"failed to import read_gef.py: {error}")

        border = _relative_border([(-2, -2), (3, -2), (3, 4), (-2, 4)])
        decoded = module.decode_border_points(border)

        self.assertEqual(decoded.tolist(), [[-2, -2], [3, -2], [3, 4], [-2, 4]])

    def test_build_absolute_polygon_uses_cell_centroid_as_anchor(self):
        module, error = try_load_module()
        self.assertIsNone(error, f"failed to import read_gef.py: {error}")

        cell = np.array([(0, 20, 30, 0, 0, 0, 0, 0, 0, 0)], dtype=CELL_DTYPE)[0]
        polygon = module.build_absolute_polygon(cell, _relative_border([(-2, -1), (4, -1), (4, 3)]))

        self.assertEqual(polygon.tolist(), [[18, 29], [24, 29], [24, 33]])

    def test_merge_gefs_removes_cross_file_overlaps_and_rebuilds_indices(self):
        module, error = try_load_module()
        self.assertIsNone(error, f"failed to import read_gef.py: {error}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_a = tmp_path / "a.cellbin.gef"
            input_b = tmp_path / "b.cellbin.gef"
            output = tmp_path / "merged.cellbin.gef"

            _write_mini_cellbin_gef(
                input_a,
                "A",
                [
                    {"x": 100, "y": 100, "border": [(-8, -8), (8, -8), (8, 8), (-8, 8)], "expr": {"GeneA": 2}},
                    {"x": 300, "y": 300, "border": [(-6, -6), (6, -6), (6, 6), (-6, 6)], "expr": {"GeneB": 3}},
                ],
            )
            _write_mini_cellbin_gef(
                input_b,
                "B",
                [
                    {"x": 100, "y": 100, "border": [(-8, -8), (8, -8), (8, 8), (-8, 8)], "expr": {"GeneA": 5}},
                    {"x": 450, "y": 450, "border": [(-5, -5), (5, -5), (5, 5), (-5, 5)], "expr": {"GeneC": 7}},
                ],
            )

            summary = module.merge_gefs(input_a, input_b, output, iou_threshold=0.3)

            self.assertEqual(summary["removed_cells_a"], 1)
            self.assertEqual(summary["removed_cells_b"], 1)
            self.assertEqual(summary["kept_cells"], 2)

            with h5py.File(output, "r") as handle:
                cells = handle["cellBin/cell"][:]
                self.assertEqual(len(cells), 2)
                self.assertEqual(cells["id"].tolist(), [0, 1])
                self.assertEqual(cells["x"].tolist(), [300, 450])

                cell_exp = handle["cellBin/cellExp"][:]
                for cell in cells:
                    start = int(cell["offset"])
                    stop = start + int(cell["geneCount"])
                    self.assertLessEqual(stop, len(cell_exp))

                gene = handle["cellBin/gene"][:]
                gene_exp = handle["cellBin/geneExp"][:]
                for gene_row in gene:
                    start = int(gene_row["offset"])
                    stop = start + int(gene_row["cellCount"])
                    self.assertLessEqual(stop, len(gene_exp))

                self.assertIn("codedCellBlock", handle)
                self.assertIn("L0", handle["codedCellBlock"])
                self.assertIn("info", handle["codedCellBlock"].attrs)

    def test_merge_gefs_only_removes_cross_file_duplicates(self):
        module, error = try_load_module()
        self.assertIsNone(error, f"failed to import read_gef.py: {error}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_a = tmp_path / "a.cellbin.gef"
            input_b = tmp_path / "b.cellbin.gef"
            output = tmp_path / "merged.cellbin.gef"

            _write_mini_cellbin_gef(
                input_a,
                "A",
                [
                    {"x": 100, "y": 100, "border": [(-7, -7), (7, -7), (7, 7), (-7, 7)], "expr": {"GeneA": 1}},
                    {"x": 100, "y": 100, "border": [(-7, -7), (7, -7), (7, 7), (-7, 7)], "expr": {"GeneB": 2}},
                ],
            )
            _write_mini_cellbin_gef(
                input_b,
                "B",
                [{"x": 300, "y": 300, "border": [(-5, -5), (5, -5), (5, 5), (-5, 5)], "expr": {"GeneC": 4}}],
            )

            summary = module.merge_gefs(input_a, input_b, output, iou_threshold=0.3)

            self.assertEqual(summary["removed_cells_a"], 0)
            self.assertEqual(summary["removed_cells_b"], 0)
            self.assertEqual(summary["kept_cells"], 3)

            with h5py.File(output, "r") as handle:
                self.assertEqual(len(handle["cellBin/cell"]), 3)


if __name__ == "__main__":
    unittest.main()
