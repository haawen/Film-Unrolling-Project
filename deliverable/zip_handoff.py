"""Create and verify the portable handoff ZIP from its manifest."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
HANDOFF = HERE / "handoff"
ARCHIVE = HERE / "handoff.zip"
PARTIAL = HERE / "handoff.zip.part"


def main() -> None:
    manifest_path = HANDOFF / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    paths = [Path(item["path"]) for item in files]
    if len(paths) != manifest["file_count"] or len(set(paths)) != len(paths):
        raise ValueError("Manifest file count or uniqueness mismatch")
    for path, item in zip(paths, files):
        source = (HANDOFF / path).resolve()
        if not source.is_relative_to(HANDOFF.resolve()):
            raise ValueError(f"Path escapes handoff: {path}")
        if not source.is_file() or source.stat().st_size != item["bytes"]:
            raise ValueError(f"Missing or changed handoff file: {path}")

    total = sum(item["bytes"] for item in files)
    if total != manifest["logical_bytes"]:
        raise ValueError("Manifest byte count mismatch")
    if shutil.disk_usage(HERE).free < total + 2**30:
        raise OSError("Not enough disk space for archive and 1 GiB reserve")
    if PARTIAL.exists():
        raise FileExistsError(f"Remove the old incomplete ZIP first: {PARTIAL}")

    try:
        with zipfile.ZipFile(PARTIAL, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for index, path in enumerate(paths, 1):
                archive.write(HANDOFF / path, f"handoff/{path.as_posix()}")
                if index % 100 == 0 or index == len(paths):
                    print(f"Archived {index}/{len(paths)} files", flush=True)
            archive.write(manifest_path, "handoff/MANIFEST.json")

        with zipfile.ZipFile(PARTIAL) as archive:
            expected = {f"handoff/{path.as_posix()}": item["bytes"]
                        for path, item in zip(paths, files)}
            expected["handoff/MANIFEST.json"] = manifest_path.stat().st_size
            actual = {item.filename: item.file_size for item in archive.infolist()}
            if actual != expected:
                raise ValueError("Archive entries differ from the manifest")
            if archive.read("handoff/MANIFEST.json") != manifest_path.read_bytes():
                raise ValueError("Archive manifest differs from source")
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"ZIP CRC failed: {bad}")
        os.replace(PARTIAL, ARCHIVE)
    except BaseException:
        PARTIAL.unlink(missing_ok=True)
        raise

    print(f"Verified ZIP: {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
