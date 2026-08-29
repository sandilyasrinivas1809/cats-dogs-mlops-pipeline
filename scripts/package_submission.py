"""Build the assignment submission zip.

Packages source code, tests, Git LFS config, CI/CD workflows, Dockerfile,
deployment manifests, the split manifests, and the trained model artifact.
Excludes the raw dataset, the local MLflow store, and virtualenv/cache dirs.

Usage:
    python scripts/package_submission.py [-o submission.zip]
"""

import argparse
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Individual files and whole directories to include.
INCLUDE_FILES = [
    "README.md",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
]
INCLUDE_DIRS = [
    "src",
    "tests",
    "deployment",
    ".github",
    "data/processed",
    "models",
    "scripts",
]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def iter_files() -> list[Path]:
    paths: list[Path] = []

    for name in INCLUDE_FILES:
        path = REPO_ROOT / name
        if path.is_file():
            paths.append(path)
        else:
            print(f"  warning: expected file missing, skipping: {name}")

    for name in INCLUDE_DIRS:
        directory = REPO_ROOT / name
        if not directory.is_dir():
            print(f"  warning: expected directory missing, skipping: {name}")
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not EXCLUDE_PARTS.intersection(path.parts):
                paths.append(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "submission.zip")
    args = parser.parse_args()

    files = iter_files()
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(REPO_ROOT).as_posix())

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output} ({len(files)} files, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
