import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "convert_test_tif_to_png", ROOT / "scripts" / "convert_test_tif_to_png.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConvertTestTifToPngTest(unittest.TestCase):
    def test_convert_sn_directory_rewrites_ones_to_255_and_writes_pngs(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sn_dir = root / "test" / "Y40178MC"
            sn_dir.mkdir(parents=True)

            mask = np.array([[0, 1, 2], [1, 3, 1]], dtype=np.uint8)
            image = np.array([[5, 1], [7, 9]], dtype=np.uint16)
            tifffile.imwrite(sn_dir / "sample_mask.tif", mask)
            tifffile.imwrite(sn_dir / "sample_image.tif", image)

            outputs = module.convert_tif_directory(sn_dir)

            expected_outputs = {
                sn_dir / "png" / "sample_mask.png",
                sn_dir / "png" / "sample_image.png",
            }
            self.assertEqual(set(outputs), expected_outputs)

            mask_png = cv2.imread(str(sn_dir / "png" / "sample_mask.png"), cv2.IMREAD_UNCHANGED)
            image_png = cv2.imread(str(sn_dir / "png" / "sample_image.png"), cv2.IMREAD_UNCHANGED)

            np.testing.assert_array_equal(
                mask_png,
                np.array([[0, 255, 2], [255, 3, 255]], dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                image_png,
                np.array([[5, 255], [7, 9]], dtype=np.uint16),
            )

    def test_main_accepts_input_dir(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sn_dir = root / "test" / "Y40178MC"
            sn_dir.mkdir(parents=True)
            tifffile.imwrite(
                sn_dir / "demo.tif",
                np.array([[1, 2], [3, 4]], dtype=np.uint8),
            )

            exit_code = module.main(["--input-dir", str(sn_dir)])

            self.assertEqual(exit_code, 0)
            output = cv2.imread(str(sn_dir / "png" / "demo.png"), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(
                output,
                np.array([[255, 2], [3, 4]], dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
