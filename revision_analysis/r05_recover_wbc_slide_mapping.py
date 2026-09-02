#!/usr/bin/env python3
"""Recover the official WBC slide map without downloading the 2.17 GB ZIP.

The Figshare item exposes one ZIP file.  This script uses HTTP byte ranges to
read the ZIP central directory and extract only ``labels.csv`` and
``slide_number.csv``.  It then validates the map against the natural numeric
folder ordering used to build the frozen WBC stack cache.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import struct
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "06_analysis_outputs/statistical_inference/wbc_slide_mapping"
DATASET_ROOT = Path("/mnt/d/New_folder/datasets/WBC_dataset1")
ARTICLE_API = "https://api.figshare.com/v2/articles/26781052"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/48650791"
EXPECTED_STACKS = 25_773
EXPECTED_SLIDES = 214


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "jimaging-4524210-revision/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_range(start: int | None, end: int | None) -> tuple[bytes, dict[str, str]]:
    if start is None:
        range_value = f"bytes=-{end}"
    else:
        range_value = f"bytes={start}-{end}"
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={"Range": range_value, "User-Agent": "jimaging-4524210-revision/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read(), dict(response.headers)


def central_directory() -> list[dict]:
    tail, _ = fetch_range(None, 131_072)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    eocd = struct.unpack("<4s4H2LH", tail[eocd_at : eocd_at + 22])
    directory_size, directory_offset = int(eocd[5]), int(eocd[6])
    directory, _ = fetch_range(directory_offset, directory_offset + directory_size - 1)
    entries: list[dict] = []
    position = 0
    while position + 46 <= len(directory) and directory[position : position + 4] == b"PK\x01\x02":
        values = struct.unpack("<4s6H3L5H2L", directory[position : position + 46])
        name_length, extra_length, comment_length = values[10], values[11], values[12]
        name = directory[position + 46 : position + 46 + name_length].decode("utf-8", errors="replace")
        entries.append(
            {
                "name": name,
                "compression_method": int(values[4]),
                "crc32": int(values[7]),
                "compressed_size": int(values[8]),
                "uncompressed_size": int(values[9]),
                "local_header_offset": int(values[16]),
            }
        )
        position += 46 + name_length + extra_length + comment_length
    return entries


def extract(entry: dict) -> bytes:
    offset = entry["local_header_offset"]
    header, _ = fetch_range(offset, offset + 4095)
    values = struct.unpack("<4s5H3L2H", header[:30])
    if values[0] != b"PK\x03\x04":
        raise RuntimeError(f"invalid local header for {entry['name']}")
    name_length, extra_length = int(values[-2]), int(values[-1])
    data_start = offset + 30 + name_length + extra_length
    compressed, _ = fetch_range(data_start, data_start + entry["compressed_size"] - 1)
    if entry["compression_method"] == 0:
        raw = compressed
    elif entry["compression_method"] == 8:
        raw = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported ZIP method {entry['compression_method']}")
    if len(raw) != entry["uncompressed_size"]:
        raise RuntimeError(f"size mismatch for {entry['name']}")
    if (zlib.crc32(raw) & 0xFFFFFFFF) != entry["crc32"]:
        raise RuntimeError(f"CRC mismatch for {entry['name']}")
    return raw


def natural_numeric_folders() -> list[int]:
    values = []
    for path in DATASET_ROOT.iterdir():
        if path.is_dir() and path.name.isdigit():
            values.append(int(path.name))
    return sorted(values)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    article = fetch_json(ARTICLE_API)
    entries = central_directory()
    wanted = {
        "labels.csv": "multi-focus-wbc-dataset/labels.csv",
        "slide_number.csv": "multi-focus-wbc-dataset/slide_number.csv",
    }
    extracted: dict[str, bytes] = {}
    for output_name, archive_name in wanted.items():
        matches = [entry for entry in entries if entry["name"] == archive_name]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {archive_name}; found {len(matches)}")
        extracted[output_name] = extract(matches[0])
        (OUT / output_name).write_bytes(extracted[output_name])

    slide_rows = list(csv.DictReader(io.StringIO(extracted["slide_number.csv"].decode("utf-8-sig"))))
    label_rows = list(csv.DictReader(io.StringIO(extracted["labels.csv"].decode("utf-8-sig"))))
    image_numbers = [int(row["img_num"]) for row in slide_rows]
    slide_numbers = [int(row["slide_num"]) for row in slide_rows]
    folders = natural_numeric_folders()
    label_numbers = [int(row["img_num"]) for row in label_rows]
    mapping = [
        {
            "stack_index": index,
            "img_num": image_number,
            "source_folder": str(DATASET_ROOT / str(image_number)),
            "slide_num": slide_number,
        }
        for index, (image_number, slide_number) in enumerate(zip(image_numbers, slide_numbers))
    ]
    with (OUT / "wbc_stack_to_slide.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)

    checks = {
        "rows": len(slide_rows),
        "label_rows": len(label_rows),
        "unique_images": len(set(image_numbers)),
        "unique_slides": len(set(slide_numbers)),
        "image_numbers_complete_zero_based": image_numbers == list(range(EXPECTED_STACKS)),
        "labels_complete_zero_based": label_numbers == list(range(EXPECTED_STACKS)),
        "local_numeric_folders_match": folders == image_numbers,
        "expected_stack_count": len(slide_rows) == EXPECTED_STACKS,
        "expected_slide_count": len(set(slide_numbers)) == EXPECTED_SLIDES,
        "patient_mapping_available": False,
    }
    status = "PASS" if all(value for key, value in checks.items() if key != "patient_mapping_available") else "FAIL"
    provenance = {
        "generated_at": now(),
        "status": status,
        "figshare_article": article.get("figshare_url"),
        "figshare_doi": article.get("doi"),
        "figshare_file_id": 48650791,
        "figshare_file_md5": article["files"][0]["computed_md5"],
        "download_strategy": "HTTP byte-range extraction of two CSV members from official ZIP",
        "mapping_basis": "natural numeric folder order used by frozen dataset loader equals img_num 0..25772",
        "patient_mapping_note": "The official CSV maps cells to 214 slides but does not map slides to the 72 patients.",
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in extracted.items()},
        "checks": checks,
    }
    (OUT / "validation_summary.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (OUT / "WBC_SLIDE_MAPPING_FINDINGS.md").write_text(
        "# WBC slide mapping findings\n\n"
        f"- Status: **{status}**.\n"
        f"- Recovered {len(slide_rows):,} authoritative cell-to-slide rows covering {len(set(slide_numbers))} slides.\n"
        "- The numeric image identifiers exactly match the natural folder order used by the frozen cache builder.\n"
        "- The release does not include a slide-to-patient mapping; patient-level resampling remains unavailable.\n"
        "- Slide-level clustered resampling is therefore the highest authoritative WBC unit available.\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
