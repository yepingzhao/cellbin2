import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_GEF = ROOT / "test" / "Y40178MC" / "Y40178MC_Transcriptomics.cellbin.gef"


def load_module():
    spec = importlib.util.spec_from_file_location("read_gef", ROOT / "read_gef.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReadGefTest(unittest.TestCase):
    def test_module_has_no_stereo_backend(self):
        module = load_module()
        self.assertFalse(hasattr(module, "read_with_stereo"))

    def test_summarize_cell_bin_gef(self):
        module = load_module()
        summary = module.summarize_gef(SAMPLE_GEF)

        self.assertEqual(summary["format"], "cell_bin")
        self.assertEqual(summary["cell_count"], 11648)
        self.assertEqual(summary["gene_count"], 51817)
        self.assertEqual(summary["expression_count"], 31528)


if __name__ == "__main__":
    unittest.main()
