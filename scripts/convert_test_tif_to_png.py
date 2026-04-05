#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np
import tifffile


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for TIFF-to-PNG conversion."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert TIFF images under test/{SN} to PNG, replacing pixel value 1 with 255."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sn", help="Sample name under the test directory, such as Y40178MC.")
    group.add_argument(
        "--input-dir",
        type=Path,
        help="Direct path to a sample directory that contains TIFF files.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("test"),
        help="Root directory that contains sample folders. Default: test",
    )
    return parser.parse_args(argv)


def resolve_input_dir(args: argparse.Namespace) -> Path:
    """Resolve the sample directory from CLI arguments."""
    if args.input_dir is not None:
        return args.input_dir.resolve()
    return (args.test_root / args.sn).resolve()


def iter_tif_files(input_dir: Path) -> Iterable[Path]:
    """Yield TIFF files directly under the sample directory."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def rewrite_foreground_values(image: np.ndarray) -> np.ndarray:
    """Return an image copy where every pixel value of 1 becomes 255."""
    updated = np.squeeze(image).copy()
    updated[updated == 1] = 255
    return updated


def validate_png_compatibility(image: np.ndarray, source: Path) -> None:
    """Check that the converted array can be written as a PNG."""
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported TIFF shape for PNG output: {source} -> {image.shape}")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported channel count for PNG output: {source} -> {image.shape}")


def convert_tif_file(input_path: Path, output_dir: Path) -> Path:
    """Convert one TIFF file into a PNG under the output directory."""
    image = tifffile.imread(input_path)
    converted = rewrite_foreground_values(image)
    validate_png_compatibility(converted, input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.png"
    success = cv2.imwrite(str(output_path), converted)
    if not success:
        raise RuntimeError(f"Failed to write PNG file: {output_path}")
    return output_path


def convert_tif_directory(input_dir: Path) -> List[Path]:
    """Convert all TIFF files directly under the sample directory."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    tif_files = list(iter_tif_files(input_dir))
    if not tif_files:
        raise FileNotFoundError(f"No TIFF files found under: {input_dir}")

    output_dir = input_dir / "png"
    return [convert_tif_file(path, output_dir) for path in tif_files]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for converting sample TIFF files to PNG."""
    args = parse_args(argv)
    input_dir = resolve_input_dir(args)
    outputs = convert_tif_directory(input_dir)

    print(f"Converted {len(outputs)} TIFF file(s) from {input_dir} to {input_dir / 'png'}")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
