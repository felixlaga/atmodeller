# SPDX-License-Identifier: Apache-2.0

"""Verify the imported investigation without rerunning or overwriting its results."""

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent / "hydrogen_storage"
    manifest = json.loads((root / "import_manifest.json").read_text())
    for name, original in manifest["original_sha256"].items():
        expected = manifest["path_changes"].get(name, {}).get("imported_sha256", original)
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Imported archive checksum mismatch: {name}")
    source_root = root / "investigation" / "sources"
    downloads = json.loads((source_root / "downloaded.json").read_text())
    for item in downloads:
        actual = hashlib.sha256((source_root / item["name"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Source download checksum mismatch: {item['name']}")
    print(
        f"Verified {len(manifest['original_sha256'])} imported files and {len(downloads)} source downloads."
    )


if __name__ == "__main__":
    main()
