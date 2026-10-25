#!/usr/bin/env python3
"""Download the pinned KION archive, verify SHA-256 and extract three CSV tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


COMMIT = "f69775be31fa5779907cf0a92ddedb70037fb5ae"
URL = f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{COMMIT}/data_original.zip"
SHA256 = "50fc5b7db78cbda4c3d0ee5bb5bea4c416cff41e8af58aab0b661d0d3e9dfa9b"
EXPECTED = ("interactions.csv", "items.csv", "users.csv")
CHUNK_BYTES = 1024 * 1024


def verify_checksum(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != SHA256:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {SHA256}, got {actual}. No data extracted.")


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "kion-temporal-recsys/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            downloaded = 0
            for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded % (32 * CHUNK_BYTES) < CHUNK_BYTES:
                    print(f"Downloaded {downloaded / CHUNK_BYTES:.0f} MiB", file=sys.stderr)
            output.flush()
            os.fsync(output.fileno())
        verify_checksum(temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def extract_tables(archive: Path, output: Path) -> dict[str, int]:
    """Copy only exact allowlisted members; never call ZipFile.extractall."""
    target = output / "data_original"
    target.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    with zipfile.ZipFile(archive) as bundle:
        manifest = bundle.infolist()
        members = {}
        for name in EXPECTED:
            matches = [entry for entry in manifest if entry.filename == f"data_original/{name}"]
            if len(matches) != 1:
                raise ValueError(f"Archive must contain exactly one data_original/{name}")
            info = matches[0]
            mode = info.external_attr >> 16
            if info.is_dir() or (mode & 0o170000) == 0o120000:
                raise ValueError(f"Expected a regular CSV member, got {info.filename}")
            if not 0 < info.file_size <= 1024**3:
                raise ValueError(f"Unexpected size for {info.filename}: {info.file_size}")
            members[name] = info
        # Destination filenames are literals from EXPECTED, never ZIP paths.
        for name, info in members.items():
            destination = target / name
            temporary = destination.with_name(destination.name + ".part")
            try:
                with bundle.open(info, "r") as source, temporary.open("wb") as stream:
                    shutil.copyfileobj(source, stream, length=CHUNK_BYTES)
                    stream.flush()
                    os.fsync(stream.fileno())
                if temporary.stat().st_size != info.file_size:
                    raise ValueError(f"Extracted size differs from manifest for {name}")
                temporary.replace(destination)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            sizes[name] = info.file_size
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--data-dir", dest="output", type=Path, default=Path("data/raw"),
                        help="Directory receiving data_original/*.csv (default: data/raw)")
    parser.add_argument("--archive", type=Path,
                        help="Verify and extract this local archive without any network request")
    args = parser.parse_args()
    archive = args.archive if args.archive is not None else args.output / "data_original.zip"
    if args.archive is not None and not archive.is_file():
        parser.error(f"Local archive does not exist: {archive}")
    if archive.is_file():
        verify_checksum(archive)
        print(f"Verified existing archive: {archive}", file=sys.stderr)
    else:
        download_archive(archive)
    sizes = extract_tables(archive, args.output)
    print(json.dumps({"source_url": URL, "commit": COMMIT, "sha256": SHA256,
                      "archive": str(archive.resolve()), "output": str(args.output.resolve()),
                      "extracted_bytes": sizes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
