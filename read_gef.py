from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import h5py
import numpy as np


DEFAULT_GEF = Path("test/Y40178MC/Y40178MC_Transcriptomics.cellbin.gef")


def _to_builtin(value: Any) -> Any:
    """Convert numpy and HDF5 values into JSON-serializable Python types."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _to_builtin(value[()])
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


def _attr_scalar(attrs: h5py.AttributeManager, key: str, default: Any = None) -> Any:
    if key not in attrs:
        return default
    value = _to_builtin(attrs[key])
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _record_to_dict(record: np.void) -> Dict[str, Any]:
    if record.dtype.names is None:
        return {"value": _to_builtin(record)}
    return {name: _to_builtin(record[name]) for name in record.dtype.names}


def _dataset_preview(dataset: h5py.Dataset, limit: int) -> List[Any]:
    if limit <= 0 or dataset.shape == () or dataset.shape[0] == 0:
        return []

    sample = dataset[: min(limit, dataset.shape[0])]
    if dataset.dtype.names is None:
        return [_to_builtin(item) for item in sample]
    return [_record_to_dict(item) for item in sample]


def detect_gef_format(file_path: Union[str, Path]) -> str:
    """Detect whether a GEF file contains cell-bin or binned expression data."""
    with h5py.File(file_path, "r") as handle:
        if "cellBin" in handle:
            return "cell_bin"
        if "geneExp" in handle:
            return "binned_gef"
    raise ValueError(f"Unsupported GEF layout: {file_path}")


def _summarize_cell_bin(handle: h5py.File, file_path: Path, preview: int) -> Dict[str, Any]:
    cell_dataset = handle["cellBin/cell"]
    gene_dataset = handle["cellBin/gene"]
    cell_exp_dataset = handle.get("cellBin/cellExp")

    return {
        "path": str(file_path),
        "format": "cell_bin",
        "cell_count": int(cell_dataset.shape[0]),
        "gene_count": int(gene_dataset.shape[0]),
        "expression_count": int(cell_exp_dataset.shape[0]) if cell_exp_dataset is not None else 0,
        "x_range": [
            _attr_scalar(cell_dataset.attrs, "minX"),
            _attr_scalar(cell_dataset.attrs, "maxX"),
        ],
        "y_range": [
            _attr_scalar(cell_dataset.attrs, "minY"),
            _attr_scalar(cell_dataset.attrs, "maxY"),
        ],
        "cell_metrics": {
            "average_area": _attr_scalar(cell_dataset.attrs, "averageArea"),
            "average_gene_count": _attr_scalar(cell_dataset.attrs, "averageGeneCount"),
            "average_exp_count": _attr_scalar(cell_dataset.attrs, "averageExpCount"),
            "average_dnb_count": _attr_scalar(cell_dataset.attrs, "averageDnbCount"),
            "median_area": _attr_scalar(cell_dataset.attrs, "medianArea"),
            "median_gene_count": _attr_scalar(cell_dataset.attrs, "medianGeneCount"),
            "median_exp_count": _attr_scalar(cell_dataset.attrs, "medianExpCount"),
            "median_dnb_count": _attr_scalar(cell_dataset.attrs, "medianDnbCount"),
        },
        "cell_preview": _dataset_preview(cell_dataset, preview),
        "gene_preview": _dataset_preview(gene_dataset, preview),
    }


def _iter_bin_names(gene_exp_group: h5py.Group) -> Iterable[str]:
    def sort_key(name: str) -> int:
        if name.startswith("bin") and name[3:].isdigit():
            return int(name[3:])
        return sys.maxsize

    return sorted(gene_exp_group.keys(), key=sort_key)


def _summarize_binned_gef(handle: h5py.File, file_path: Path, preview: int) -> Dict[str, Any]:
    gene_exp_group = handle["geneExp"]
    bins: List[Dict[str, Any]] = []

    for bin_name in _iter_bin_names(gene_exp_group):
        expression_dataset = gene_exp_group[f"{bin_name}/expression"]
        bins.append(
            {
                "bin": bin_name,
                "expression_count": int(expression_dataset.shape[0]),
                "x_range": [
                    _attr_scalar(expression_dataset.attrs, "minX"),
                    _attr_scalar(expression_dataset.attrs, "maxX"),
                ],
                "y_range": [
                    _attr_scalar(expression_dataset.attrs, "minY"),
                    _attr_scalar(expression_dataset.attrs, "maxY"),
                ],
                "preview": _dataset_preview(expression_dataset, preview),
            }
        )

    return {
        "path": str(file_path),
        "format": "binned_gef",
        "bins": bins,
    }


def summarize_gef(file_path: Union[str, Path], preview: int = 3) -> Dict[str, Any]:
    """Read a GEF file and return a JSON-serializable summary."""
    path = Path(file_path)
    with h5py.File(path, "r") as handle:
        if "cellBin" in handle:
            return _summarize_cell_bin(handle, path, preview)
        if "geneExp" in handle:
            return _summarize_binned_gef(handle, path, preview)
    raise ValueError(f"Unsupported GEF layout: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read a GEF file and print a concise summary.")
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=DEFAULT_GEF if DEFAULT_GEF.exists() else None,
        help="Path to a .gef file. Defaults to the sample file in test/Y40178MC when available.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of rows to show in the preview section.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.file is None:
        parser.error("missing GEF file path")

    try:
        summary = summarize_gef(args.file, preview=max(args.preview, 0))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"Failed to read GEF: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
