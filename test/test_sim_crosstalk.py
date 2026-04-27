from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import errno
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import tifffile
import anndata as ad

CURR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(CURR_PATH)

import sim_crosstalk


def write_gem(path: Path, rows: list[dict[str, object]], header: str = "# synthetic gem") -> None:
    frame = pd.DataFrame(rows, columns=sim_crosstalk.GEM_COLUMNS)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{header}\n")
        frame.to_csv(handle, sep="\t", index=False)


def write_label_mask(path: Path, label_image: np.ndarray) -> None:
    tifffile.imwrite(path, label_image.astype(np.uint8))


def make_polygon_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class MixedGemTest(unittest.TestCase):
    def test_build_mixed_gem_preserves_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mc_gem = tmp_path / "mc.gem"
            p5_gem = tmp_path / "p5.gem"
            output_gem = tmp_path / "mixed.gem"
            write_gem(
                mc_gem,
                [{"geneID": "A", "x": 0, "y": 0, "MIDCount": 1, "ExonCount": 1}],
                header="# MC header",
            )
            write_gem(
                p5_gem,
                [{"geneID": "B", "x": 1, "y": 1, "MIDCount": 2, "ExonCount": 1}],
                header="# P5 header",
            )

            summary = sim_crosstalk.build_mixed_gem(mc_gem, p5_gem, output_gem)

            self.assertEqual(summary["mixed_row_count"], 2)
            self.assertEqual(summary["mixed_mid_total"], 3)
            lines = output_gem.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "# MC header")
            mixed_df = pd.read_csv(output_gem, sep="\t", comment="#")
            self.assertEqual(mixed_df["geneID"].tolist(), ["A", "B"])

    def test_convert_gem_to_gef_calls_cellbin2_converter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gem_path = tmp_path / "sample.gem"
            gef_path = tmp_path / "sample.gef"
            write_gem(
                gem_path,
                [{"geneID": "A", "x": 0, "y": 0, "MIDCount": 1, "ExonCount": 1}],
            )

            called = {}

            def fake_gem_to_gef(src_path, dst_path):
                called["src"] = src_path
                called["dst"] = dst_path
                Path(dst_path).write_text("gef", encoding="utf-8")

            with mock.patch("cellbin2.matrix.matrix.gem_to_gef", side_effect=fake_gem_to_gef):
                result = sim_crosstalk.convert_gem_to_gef(gem_path, gef_path)

            self.assertEqual(result, gef_path)
            self.assertEqual(called["src"], str(gem_path))
            self.assertEqual(called["dst"], str(gef_path))
            self.assertTrue(gef_path.exists())


class PolygonExtractionTest(unittest.TestCase):
    def test_extract_cell_polygons_handles_single_pixel_components(self) -> None:
        label_image = np.array(
            [
                [1, 0, 2],
                [0, 0, 0],
                [3, 0, 0],
            ],
            dtype=np.int32,
        )

        stats = sim_crosstalk.extract_cell_polygons(label_image, source_name="MC")

        self.assertEqual(sorted(stats["cell_label"].tolist()), ["MC_1", "MC_2", "MC_3"])
        self.assertTrue(all(str(value).startswith("POLYGON") for value in stats["contour_wkt"]))


class MappingRuleTest(unittest.TestCase):
    def test_map_mixed_cells_prefers_highest_iou_and_marks_doublets(self) -> None:
        mixed_cells = make_polygon_frame(
            [
                {
                    "cell_label": "mixed_only_mc",
                    "source": "mixed",
                    "contour_wkt": "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
                },
                {
                    "cell_label": "mixed_doublet",
                    "source": "mixed",
                    "contour_wkt": "POLYGON ((10 0, 14 0, 14 4, 10 4, 10 0))",
                },
                {
                    "cell_label": "mixed_tie",
                    "source": "mixed",
                    "contour_wkt": "POLYGON ((20 0, 22 0, 22 2, 20 2, 20 0))",
                },
            ]
        )
        mc_cells = make_polygon_frame(
            [
                {
                    "cell_label": "MC_a",
                    "source": "MC",
                    "contour_wkt": "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
                },
                {
                    "cell_label": "MC_b",
                    "source": "MC",
                    "contour_wkt": "POLYGON ((10 0, 12 0, 12 4, 10 4, 10 0))",
                },
                {
                    "cell_label": "MC_c",
                    "source": "MC",
                    "contour_wkt": "POLYGON ((20 0, 22 0, 22 2, 20 2, 20 0))",
                },
                {
                    "cell_label": "MC_d",
                    "source": "MC",
                    "contour_wkt": "POLYGON ((20 0, 22 0, 22 2, 20 2, 20 0))",
                },
            ]
        )
        p5_cells = make_polygon_frame(
            [
                {
                    "cell_label": "P5_a",
                    "source": "P5",
                    "contour_wkt": "POLYGON ((12 0, 14 0, 14 4, 12 4, 12 0))",
                }
            ]
        )

        candidates, mapping = sim_crosstalk.map_mixed_cells(
            mixed_cells=mixed_cells,
            mc_cells=mc_cells,
            p5_cells=p5_cells,
            iou_threshold=0.1,
        )

        self.assertGreaterEqual(len(candidates), 4)
        result = mapping.set_index("mixed_cell_label")
        self.assertEqual(result.loc["mixed_only_mc", "mapped_source_dataset"], "MC")
        self.assertEqual(result.loc["mixed_only_mc", "mapped_source_label"], "MC_a")
        self.assertTrue(bool(result.loc["mixed_doublet", "is_doublet"]))
        self.assertEqual(result.loc["mixed_doublet", "doublet_reason"], "cross_source_overlap")
        self.assertEqual(result.loc["mixed_tie", "mapped_source_label"], "MC_c")

    def test_filter_mixed_doublets_removes_all_rows_for_doublet_cells(self) -> None:
        molecules = pd.DataFrame(
            [
                {"geneID": "A", "x": 0, "y": 0, "MIDCount": 1, "ExonCount": 1, "cell_label": "keep_me"},
                {"geneID": "B", "x": 1, "y": 1, "MIDCount": 1, "ExonCount": 1, "cell_label": "drop_me"},
            ]
        )
        mapping = pd.DataFrame(
            [
                {"mixed_cell_label": "keep_me", "is_doublet": False},
                {"mixed_cell_label": "drop_me", "is_doublet": True},
            ]
        )

        filtered = sim_crosstalk.filter_mixed_doublets(molecules, mapping)

        self.assertEqual(filtered["cell_label"].tolist(), ["keep_me"])

    def test_build_mapped_h5ad_inputs_aligns_union_genes_and_keeps_mixed_order(self) -> None:
        mixed_cells = pd.DataFrame(
            [
                {"cell_label": "mixed_2"},
                {"cell_label": "mixed_1"},
                {"cell_label": "mixed_drop"},
            ]
        )
        filtered_molecules = pd.DataFrame(
            [
                {"cell_label": "mixed_2", "geneID": "G2", "MIDCount": 5},
                {"cell_label": "mixed_1", "geneID": "G1", "MIDCount": 3},
                {"cell_label": "mixed_1", "geneID": "G3", "MIDCount": 7},
            ]
        )
        mapping_df = pd.DataFrame(
            [
                {
                    "mixed_cell_label": "mixed_2",
                    "mapped_source_dataset": "P5",
                    "mapped_source_label": "P5_9",
                    "mapped_iou": 0.8,
                    "mapped_intersection_area": 12.0,
                    "is_doublet": False,
                    "doublet_reason": "",
                },
                {
                    "mixed_cell_label": "mixed_1",
                    "mapped_source_dataset": "MC",
                    "mapped_source_label": "MC_7",
                    "mapped_iou": 0.9,
                    "mapped_intersection_area": 15.0,
                    "is_doublet": False,
                    "doublet_reason": "",
                },
                {
                    "mixed_cell_label": "mixed_drop",
                    "mapped_source_dataset": "",
                    "mapped_source_label": "",
                    "mapped_iou": 0.0,
                    "mapped_intersection_area": 0.0,
                    "is_doublet": False,
                    "doublet_reason": "",
                },
            ]
        )
        mc_molecules = pd.DataFrame(
            [
                {"cell_label": "MC_7", "geneID": "G1", "MIDCount": 10},
                {"cell_label": "MC_7", "geneID": "G4", "MIDCount": 2},
            ]
        )
        p5_molecules = pd.DataFrame(
            [
                {"cell_label": "P5_9", "geneID": "G2", "MIDCount": 11},
                {"cell_label": "P5_9", "geneID": "G4", "MIDCount": 1},
            ]
        )

        obs_df, gene_order, mixed_x, source_x = sim_crosstalk.build_mapped_h5ad_inputs(
            mixed_cells=mixed_cells,
            filtered_molecules=filtered_molecules,
            mapping_df=mapping_df,
            mc_molecules=mc_molecules,
            p5_molecules=p5_molecules,
        )

        self.assertEqual(obs_df["mixed_cell_label"].tolist(), ["mixed_2", "mixed_1"])
        self.assertEqual(gene_order.tolist(), ["G1", "G2", "G3", "G4"])
        np.testing.assert_array_equal(mixed_x.toarray(), np.array([[0, 5, 0, 0], [3, 0, 7, 0]]))
        np.testing.assert_array_equal(source_x.toarray(), np.array([[0, 11, 0, 1], [10, 0, 0, 2]]))

    def test_build_mapped_h5ad_inputs_uses_batch_source_join_and_skips_missing_matches(self) -> None:
        class GuardedSeries(pd.Series):
            @property
            def _constructor(self):
                return GuardedSeries

            def __eq__(self, other):
                raise AssertionError(f"unexpected per-cell equality filter: {other}")

        class GuardedDataFrame(pd.DataFrame):
            @property
            def _constructor(self):
                return GuardedDataFrame

            @property
            def _constructor_sliced(self):
                return GuardedSeries

        mixed_cells = pd.DataFrame(
            [
                {"cell_label": "mixed_3"},
                {"cell_label": "mixed_2"},
                {"cell_label": "mixed_1"},
            ]
        )
        filtered_molecules = pd.DataFrame(
            [
                {"cell_label": "mixed_3", "geneID": "G5", "MIDCount": 4},
                {"cell_label": "mixed_2", "geneID": "G2", "MIDCount": 5},
                {"cell_label": "mixed_1", "geneID": "G1", "MIDCount": 3},
            ]
        )
        mapping_df = pd.DataFrame(
            [
                {
                    "mixed_cell_label": "mixed_3",
                    "mapped_source_dataset": "MC",
                    "mapped_source_label": "MC_missing",
                    "mapped_iou": 0.7,
                    "mapped_intersection_area": 10.0,
                    "is_doublet": False,
                    "doublet_reason": "",
                },
                {
                    "mixed_cell_label": "mixed_2",
                    "mapped_source_dataset": "P5",
                    "mapped_source_label": "P5_9",
                    "mapped_iou": 0.8,
                    "mapped_intersection_area": 12.0,
                    "is_doublet": False,
                    "doublet_reason": "",
                },
                {
                    "mixed_cell_label": "mixed_1",
                    "mapped_source_dataset": "MC",
                    "mapped_source_label": "MC_7",
                    "mapped_iou": 0.9,
                    "mapped_intersection_area": 15.0,
                    "is_doublet": False,
                    "doublet_reason": "",
                },
            ]
        )
        mc_molecules = GuardedDataFrame(
            [
                {"cell_label": "MC_7", "geneID": "G1", "MIDCount": 10},
                {"cell_label": "MC_7", "geneID": "G4", "MIDCount": 2},
            ]
        )
        p5_molecules = GuardedDataFrame(
            [
                {"cell_label": "P5_9", "geneID": "G2", "MIDCount": 11},
                {"cell_label": "P5_9", "geneID": "G4", "MIDCount": 1},
            ]
        )

        obs_df, gene_order, mixed_x, source_x = sim_crosstalk.build_mapped_h5ad_inputs(
            mixed_cells=mixed_cells,
            filtered_molecules=filtered_molecules,
            mapping_df=mapping_df,
            mc_molecules=mc_molecules,
            p5_molecules=p5_molecules,
        )

        self.assertEqual(obs_df["mixed_cell_label"].tolist(), ["mixed_3", "mixed_2", "mixed_1"])
        self.assertEqual(gene_order.tolist(), ["G1", "G2", "G4", "G5"])
        np.testing.assert_array_equal(mixed_x.toarray(), np.array([[0, 0, 0, 4], [0, 5, 0, 0], [3, 0, 0, 0]]))
        np.testing.assert_array_equal(source_x.toarray(), np.array([[0, 0, 0, 0], [0, 11, 1, 0], [10, 0, 2, 0]]))


class MainFlowTest(unittest.TestCase):
    def test_main_runs_cellbin2_pipeline_with_mocked_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mc_gem = tmp_path / "mc.gem"
            p5_gem = tmp_path / "p5.gem"
            template = tmp_path / "template.json"
            output_dir = tmp_path / "out"

            write_gem(
                mc_gem,
                [
                    {"geneID": "A", "x": 0, "y": 0, "MIDCount": 1, "ExonCount": 1},
                    {"geneID": "B", "x": 1, "y": 0, "MIDCount": 1, "ExonCount": 1},
                ],
            )
            write_gem(
                p5_gem,
                [
                    {"geneID": "C", "x": 10, "y": 0, "MIDCount": 1, "ExonCount": 1},
                    {"geneID": "D", "x": 11, "y": 0, "MIDCount": 1, "ExonCount": 1},
                ],
            )
            template.write_text(
                json.dumps(
                    {
                        "image_process": {
                            "0": {
                                "file_path": "",
                                "tech_type": "Transcriptomics",
                                "chip_detect": False,
                                "quality_control": False,
                                "tissue_segmentation": False,
                                "cell_segmentation": True,
                                "channel_align": -1,
                                "magnification": 10,
                                "registration": {"fixed_image": -1, "trackline": False, "reuse": -1},
                                "chip_matching": -1,
                                "tissue_filter": -1,
                            }
                        },
                        "molecular_classify": {
                            "0": {
                                "exp_matrix": 0,
                                "cell_mask": {"nuclei": [0], "interior": [], "boundary": [], "matrix": []},
                                "correct_r": 10,
                                "extra_method": "",
                            }
                        },
                        "run": {
                            "qc": False,
                            "alignment": True,
                            "matrix_extract": True,
                            "report": False,
                            "annotation": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            recorded_runs: list[tuple[str, Path, Path]] = []

            def fake_run_cellbin2(
                sample_name: str,
                config_path: Path,
                run_output_dir: Path,
                cellbin2_python: Path,
                cellbin2_entry: Path,
            ) -> dict[str, str]:
                recorded_runs.append((sample_name, config_path, run_output_dir))
                run_output_dir.mkdir(parents=True, exist_ok=True)
                label_image = np.zeros((3, 12), dtype=np.uint8)
                if sample_name == "Y40178MC":
                    label_image[0, 0:2] = 1
                elif sample_name == "Y40178P5":
                    label_image[0, 10:12] = 2
                else:
                    label_image[0, 0:2] = 1
                    label_image[0, 10:12] = 2
                mask_path = run_output_dir / f"{sample_name}_cell_mask.tif"
                write_label_mask(mask_path, label_image)
                return {"cell_mask": str(mask_path)}

            original_runner = sim_crosstalk.run_cellbin2
            try:
                sim_crosstalk.run_cellbin2 = fake_run_cellbin2
                exit_code = sim_crosstalk.main(
                    [
                        "--mc-gem",
                        str(mc_gem),
                        "--p5-gem",
                        str(p5_gem),
                        "--output-dir",
                        str(output_dir),
                        "--cellbin2-template",
                        str(template),
                        "--force",
                    ]
                )
            finally:
                sim_crosstalk.run_cellbin2 = original_runner

            self.assertEqual(exit_code, 0)
            self.assertEqual([run[0] for run in recorded_runs], ["Y40178MC", "Y40178P5", "mixed"])

            raw_mixed = pd.read_csv(output_dir / "mixed.gem", sep="\t", comment="#")
            filtered_mixed = pd.read_csv(output_dir / "mixed_filtered.gem", sep="\t", comment="#")
            mapping = pd.read_parquet(output_dir / "mixed_cell_mapping.parquet")
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            mapped_h5ad = ad.read_h5ad(output_dir / "mixed_maped.h5ad")

            self.assertEqual(len(raw_mixed), 4)
            self.assertEqual(len(filtered_mixed), 4)
            self.assertEqual(sorted(mapping["mixed_cell_label"].tolist()), ["mixed_1", "mixed_2"])
            self.assertEqual(summary["mixed"]["segmented_cell_count"], 2)
            self.assertEqual(summary["mixed"]["doublet_cell_count"], 0)
            self.assertEqual(summary["mixed"]["filtered_row_count"], 4)
            self.assertEqual(mapped_h5ad.shape[0], 2)
            self.assertIn("source", mapped_h5ad.layers.keys())
            self.assertEqual(list(mapped_h5ad.obs["mixed_cell_label"]), ["mixed_1", "mixed_2"])
            self.assertEqual(mapped_h5ad.X.shape, mapped_h5ad.layers["source"].shape)

    def test_main_reuses_existing_mixed_gem_gef_and_cellbin2_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mc_gem = tmp_path / "mc.gem"
            p5_gem = tmp_path / "p5.gem"
            template = tmp_path / "template.json"
            output_dir = tmp_path / "out"
            cellbin2_dir = output_dir / "cellbin2"

            write_gem(
                mc_gem,
                [
                    {"geneID": "A", "x": 0, "y": 0, "MIDCount": 1, "ExonCount": 1},
                ],
            )
            write_gem(
                p5_gem,
                [
                    {"geneID": "B", "x": 10, "y": 0, "MIDCount": 1, "ExonCount": 1},
                ],
            )
            template.write_text(
                json.dumps(
                    {
                        "image_process": {"0": {"file_path": "", "tech_type": "Transcriptomics"}},
                        "molecular_classify": {"0": {"exp_matrix": 0, "cell_mask": {"nuclei": [0]}}},
                        "run": {"qc": False, "alignment": True, "matrix_extract": True, "report": False, "annotation": False},
                    }
                ),
                encoding="utf-8",
            )

            sim_crosstalk.ensure_output_dir(output_dir, force=False)
            mixed_gem_path = output_dir / "mixed.gem"
            write_gem(
                mixed_gem_path,
                [
                    {"geneID": "A", "x": 0, "y": 0, "MIDCount": 1, "ExonCount": 1},
                    {"geneID": "B", "x": 10, "y": 0, "MIDCount": 1, "ExonCount": 1},
                ],
            )
            for gef_name in ("Y40178MC.gef", "Y40178P5.gef", "mixed.gef"):
                (output_dir / gef_name).write_text("existing gef", encoding="utf-8")

            label_image = np.zeros((3, 12), dtype=np.uint8)
            label_image[0, 0:2] = 1
            label_image[0, 10:12] = 2
            for sample_name in ("Y40178MC", "Y40178P5", "mixed"):
                run_dir = cellbin2_dir / sample_name
                run_dir.mkdir(parents=True, exist_ok=True)
                write_label_mask(run_dir / f"{sample_name}_cell_mask.tif", label_image)

            original_build_mixed_gem = sim_crosstalk.build_mixed_gem
            original_convert_gem_to_gef = sim_crosstalk.convert_gem_to_gef
            original_run_cellbin2 = sim_crosstalk.run_cellbin2

            def fail_build_mixed_gem(*args, **kwargs):
                raise AssertionError("build_mixed_gem should not run when mixed.gem already exists")

            def fail_convert_gem_to_gef(*args, **kwargs):
                raise AssertionError("convert_gem_to_gef should not run when GEF already exists")

            def fail_run_cellbin2(*args, **kwargs):
                raise AssertionError("run_cellbin2 should not run when cell mask already exists")

            try:
                sim_crosstalk.build_mixed_gem = fail_build_mixed_gem
                sim_crosstalk.convert_gem_to_gef = fail_convert_gem_to_gef
                sim_crosstalk.run_cellbin2 = fail_run_cellbin2

                exit_code = sim_crosstalk.main(
                    [
                        "--mc-gem",
                        str(mc_gem),
                        "--p5-gem",
                        str(p5_gem),
                        "--output-dir",
                        str(output_dir),
                        "--cellbin2-template",
                        str(template),
                    ]
                )
            finally:
                sim_crosstalk.build_mixed_gem = original_build_mixed_gem
                sim_crosstalk.convert_gem_to_gef = original_convert_gem_to_gef
                sim_crosstalk.run_cellbin2 = original_run_cellbin2

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mixed"]["row_count"], 2)
            self.assertEqual(summary["mixed"]["total_mid_count"], 2)

    def test_main_prints_progress_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mc_gem = tmp_path / "mc.gem"
            p5_gem = tmp_path / "p5.gem"
            template = tmp_path / "template.json"
            output_dir = tmp_path / "out"

            write_gem(
                mc_gem,
                [{"geneID": "A", "x": 0, "y": 0, "MIDCount": 1, "ExonCount": 1}],
            )
            write_gem(
                p5_gem,
                [{"geneID": "B", "x": 10, "y": 0, "MIDCount": 1, "ExonCount": 1}],
            )
            template.write_text(
                json.dumps(
                    {
                        "image_process": {"0": {"file_path": "", "tech_type": "Transcriptomics"}},
                        "molecular_classify": {"0": {"exp_matrix": 0, "cell_mask": {"nuclei": [0]}}},
                        "run": {"qc": False, "alignment": True, "matrix_extract": True, "report": False, "annotation": False},
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_cellbin2(
                sample_name: str,
                config_path: Path,
                run_output_dir: Path,
                cellbin2_python: Path,
                cellbin2_entry: Path,
            ) -> dict[str, str]:
                run_output_dir.mkdir(parents=True, exist_ok=True)
                label_image = np.zeros((3, 12), dtype=np.uint8)
                if sample_name == "Y40178MC":
                    label_image[0, 0:2] = 1
                elif sample_name == "Y40178P5":
                    label_image[0, 10:12] = 2
                else:
                    label_image[0, 0:2] = 1
                    label_image[0, 10:12] = 2
                mask_path = run_output_dir / f"{sample_name}_cell_mask.tif"
                write_label_mask(mask_path, label_image)
                return {"cell_mask": str(mask_path)}

            original_runner = sim_crosstalk.run_cellbin2
            stdout_buffer = StringIO()
            try:
                sim_crosstalk.run_cellbin2 = fake_run_cellbin2
                with mock.patch("sys.stdout", stdout_buffer):
                    exit_code = sim_crosstalk.main(
                        [
                            "--mc-gem",
                            str(mc_gem),
                            "--p5-gem",
                            str(p5_gem),
                            "--output-dir",
                            str(output_dir),
                            "--cellbin2-template",
                            str(template),
                            "--force",
                        ]
                    )
            finally:
                sim_crosstalk.run_cellbin2 = original_runner

            self.assertEqual(exit_code, 0)
            output = stdout_buffer.getvalue()
            self.assertIn("using output directory", output)
            self.assertIn("building mixed.gem", output)
            self.assertIn("converting GEM to GEF", output)
            self.assertIn("running CellBin2 for Y40178MC", output)
            self.assertIn("assigning molecules to segmented cells", output)
            self.assertIn("writing mixed_maped.h5ad", output)
            self.assertIn("pipeline complete", output)


class NotebookFlowTest(unittest.TestCase):
    def test_notebook_expands_pipeline_steps_instead_of_calling_main(self) -> None:
        notebook = json.loads(Path("sim_crosstalk.ipynb").read_text(encoding="utf-8"))
        joined_sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertNotIn("sim_crosstalk.main(", joined_sources)
        self.assertIn("import importlib", joined_sources)
        self.assertIn("sim_crosstalk = importlib.reload(sim_crosstalk)", joined_sources)
        self.assertIn("sim_crosstalk.build_cellbin2_env()", joined_sources)
        self.assertIn("LD_LIBRARY_PATH", joined_sources)
        self.assertIn(".resolve()", joined_sources)
        self.assertIn("if not mixed_gem_path.exists()", joined_sources)
        self.assertIn("sim_crosstalk.convert_gem_to_gef", joined_sources)
        self.assertIn("\"mixed\": mixed_gef_path", joined_sources)
        self.assertIn("force_rerun = False", joined_sources)
        self.assertIn("sim_crosstalk.ensure_output_dir(output_dir, force=force_rerun)", joined_sources)
        self.assertIn("if sample_name not in run_outputs", joined_sources)
        self.assertIn("sim_crosstalk.get_run_output", joined_sources)
        for helper_name in [
            "sim_crosstalk.build_mixed_gem",
            "sim_crosstalk.prepare_cellbin2_config",
            "sim_crosstalk.run_cellbin2",
            "sim_crosstalk.load_final_cell_labels",
            "sim_crosstalk.extract_cell_polygons",
            "sim_crosstalk.assign_molecules_to_cells",
            "sim_crosstalk.map_mixed_cells",
            "sim_crosstalk.filter_mixed_doublets",
            "sim_crosstalk.write_filtered_mixed_outputs",
            "sim_crosstalk.write_summary",
        ]:
            self.assertIn(helper_name, joined_sources)


class CellBin2RuntimeTest(unittest.TestCase):
    def test_run_cellbin2_injects_detected_nvidia_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            run_dir = tmp_path / "run"
            mask_path = run_dir / "SAMPLE_cell_mask.tif"

            fake_libs = ["/fake/cudnn/lib", "/fake/cublas/lib"]

            def fake_subprocess_run(command, check, env=None):
                self.assertEqual(command[0], sys.executable)
                self.assertTrue(env is not None)
                self.assertIn("/fake/cudnn/lib", env["LD_LIBRARY_PATH"])
                self.assertIn("/fake/cublas/lib", env["LD_LIBRARY_PATH"])
                run_dir.mkdir(parents=True, exist_ok=True)
                write_label_mask(mask_path, np.array([[1]], dtype=np.uint8))
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(sim_crosstalk, "discover_onnxruntime_gpu_lib_paths", return_value=fake_libs):
                with mock.patch.object(sim_crosstalk.subprocess, "run", side_effect=fake_subprocess_run):
                    result = sim_crosstalk.run_cellbin2(
                        sample_name="SAMPLE",
                        config_path=config_path,
                        run_output_dir=run_dir,
                        cellbin2_python=Path(sys.executable),
                        cellbin2_entry=Path("cellbin2/cellbin_pipeline.py"),
                    )

            self.assertEqual(result["cell_mask"], str(mask_path))

    def test_run_cellbin2_accepts_nonzero_exit_when_cell_mask_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            run_dir = tmp_path / "run"
            mask_path = run_dir / "SAMPLE_cell_mask.tif"

            def fake_subprocess_run(command, check, env=None):
                run_dir.mkdir(parents=True, exist_ok=True)
                write_label_mask(mask_path, np.array([[1]], dtype=np.uint8))
                return subprocess.CompletedProcess(command, 255)

            with mock.patch.object(sim_crosstalk.subprocess, "run", side_effect=fake_subprocess_run):
                result = sim_crosstalk.run_cellbin2(
                    sample_name="SAMPLE",
                    config_path=config_path,
                    run_output_dir=run_dir,
                    cellbin2_python=Path(sys.executable),
                    cellbin2_entry=Path("cellbin2/cellbin_pipeline.py"),
                )

            self.assertEqual(result["cell_mask"], str(mask_path))
            self.assertEqual(result["returncode"], "255")


class OutputDirTest(unittest.TestCase):
    def test_ensure_output_dir_retries_transient_directory_not_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir()

            calls = {"count": 0}
            real_rmtree = sim_crosstalk.shutil.rmtree

            def flaky_rmtree(path):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise OSError(errno.ENOTEMPTY, "Directory not empty")
                return real_rmtree(path)

            with mock.patch.object(sim_crosstalk.shutil, "rmtree", side_effect=flaky_rmtree):
                sim_crosstalk.ensure_output_dir(output_dir, force=True)

            self.assertTrue(output_dir.exists())
            self.assertEqual(calls["count"], 2)


if __name__ == "__main__":
    unittest.main()
