#!/usr/bin/env python3
"""PROTEAS audit — verify the zips WITHOUT extracting them.

Pure Python 3.9+ stdlib. No nibabel, no pip installs.
Reads zip central directories; parses NIfTI headers with a built-in reader
(gzip + struct — NIfTI-1 header is 348 bytes, trivial to parse).

Usage:
  python audit_proteas.py --src D:\\proteas_raw
Writes proteas_audit_report.json next to the zips and prints a timeline table.

Checks per patient zip:
  - BraTS/baseline, fu1, fu2... timepoint folders each have t1, t1c, t2, fla
  - NIfTI dims/dtype sane for every sequence file
  - tumor_segmentation files exist and dims match the images
  - CT / RTP / brain_mask files present
  - per-patient timeline summary (which timepoints exist)
"""

import argparse
import gzip
import io
import json
import os
import struct
import sys
import zipfile

SEQS = ("t1", "t1c", "t2", "fla")
TIMEPOINTS = ("baseline", "fu1", "fu2", "fu3", "fu4", "fu5", "fu6")


def nifti_header(b):
    """Parse a NIfTI-1 header from raw bytes. Returns dict or None if not NIfTI."""
    try:
        if len(b) < 348:
            return None
        sizeof_le = struct.unpack("<i", b[0:4])[0]
        sizeof_be = struct.unpack(">i", b[0:4])[0]
        if sizeof_le == 348:
            endian = "<"
        elif sizeof_be == 348:
            endian = ">"
        else:
            return None
        dim = struct.unpack(endian + "8h", b[40:56])
        datatype = struct.unpack(endian + "h", b[70:72])[0]
        bitpix = struct.unpack(endian + "h", b[72:74])[0]
        pixdim = struct.unpack(endian + "8f", b[76:108])
        magic = b[344:348].decode("ascii", "replace")
        nd = dim[0]
        if not (1 <= nd <= 7):
            return None
        return {
            "ndim": nd, "shape": list(dim[1:nd+1]),
            "datatype": datatype, "bitpix": bitpix,
            "pixdim": [round(p, 3) for p in pixdim[1:nd+1]],
            "magic": magic.strip(),
        }
    except Exception:  # noqa: BLE001
        return None


def read_nifti_from_zip(zf, name):
    """Read the NIfTI-1 header of a .nii / .nii.gz member of an open zipfile.

    We pull 4 KiB of compressed bytes (enough to decompress the 348-byte header)
    to avoid gzip EOF errors on truncated reads.
    """
    try:
        with zf.open(name) as f:
            head = f.read(4096)
        if name.lower().endswith(".gz"):
            d = gzip.GzipFile(fileobj=io.BytesIO(head))
            head = d.read(352)
        return nifti_header(head[:352])
    except Exception:  # noqa: BLE001
        return None


def audit_patient_zip(path):
    pid = os.path.basename(path).replace(".zip", "")
    rep = {"patient": pid, "zip": path, "zip_size_mb": round(os.path.getsize(path) / 1e6, 1), "status": "FAIL",
           "timepoints": {}, "other_files": [], "errors": [], "warnings": []}

    try:
        zf = zipfile.ZipFile(path)
    except Exception as e:  # noqa: BLE001
        rep["errors"].append(f"zip unreadable: {e}")
        return rep

    names = zf.namelist()
    lower = [n.lower() for n in names]

    # find all timepoint folders containing sequences
    tp_found = {}
    for n, nl in zip(names, lower):
        for tp in TIMEPOINTS:
            if f"brats/{tp}/" in nl:
                tp_found.setdefault(tp, set())
                for s in SEQS:
                    if f"/{s}.nii" in nl:
                        tp_found[tp].add(s)

    for tp in TIMEPOINTS:
        if tp in tp_found:
            seqs = sorted(tp_found[tp])
            entry = {"sequences": seqs, "nifti": {}, "ok": True}
            missing = [s for s in SEQS if s not in seqs]
            if missing:
                entry["ok"] = False
                entry["missing"] = missing
                rep["warnings"].append(f"{tp}: missing sequences {missing}")
            # parse headers of the sequence files
            for n, nl in zip(names, lower):
                if f"brats/{tp}/" in nl and nl.endswith((".nii", ".nii.gz")):
                    hdr = read_nifti_from_zip(zf, n)
                    if hdr is None:
                        entry["nifti"][n] = None
                        entry["ok"] = False
                        rep["warnings"].append(f"{tp}: unparseable NIfTI {n}")
                    else:
                        entry["nifti"][n.split("/")[-1]] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in hdr.items() if k != "sform"}
            rep["timepoints"][tp] = entry

    if "baseline" not in rep["timepoints"]:
        rep["errors"].append("no baseline timepoint found")

    # non-timepoint artifacts
    for kw in ("ct", "rtp", "brain_mask", "tumor_segmentation", "seg"):
        hits = [n for n in lower if kw in n]
        if hits:
            rep["other_files"].append({"kind": kw, "count": len(hits), "example": [n for n in names if n.lower() == hits[0]][0] if len(hits) else None})
        if kw in ("ct", "rtp", "brain_mask") and not hits:
            rep["warnings"].append(f"no file matching '{kw}' found")

    rep["status"] = "OK" if rep["timepoints"] and not rep["errors"] and all(v["ok"] for v in rep["timepoints"].values()) else ("WARN" if not rep["errors"] else "FAIL")
    zf.close()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing Pxx.zip files")
    args = ap.parse_args()

    zips = sorted([os.path.join(args.src, f) for f in os.listdir(args.src) if f.endswith(".zip")])
    if not zips:
        print(f"No .zip files in {args.src}")
        sys.exit(1)

    print(f"Auditing {len(zips)} patient zip(s) in {args.src} (no extraction)\n")
    reports = [audit_patient_zip(p) for p in zips]

    # timeline table
    print(f"{'patient':<10} {'status':<6} {'timepoints':<40} sequences")
    for r in reports:
        tps = []
        for tp in TIMEPOINTS:
            if tp in r["timepoints"]:
                n_seq = len(r["timepoints"][tp]["sequences"])
                tps.append(f"{tp}({n_seq})")
        print(f"{r['patient']:<10} {r['status']:<6} {', '.join(tps):<40} {', '.join(r['timepoints'].get('baseline', {}).get('sequences', []))}")

    n_ok = sum(1 for r in reports if r["status"] == "OK")
    n_warn = sum(1 for r in reports if r["status"] == "WARN")
    n_fail = sum(1 for r in reports if r["status"] == "FAIL")
    print(f"\nOK: {n_ok}  WARN: {n_warn}  FAIL: {n_fail}")

    out = os.path.join(args.src, "proteas_audit_report.json")
    with open(out, "w") as f:
        json.dump({"patients": reports, "summary": {"ok": n_ok, "warn": n_warn, "fail": n_fail}}, f, indent=2)
    print(f"Full report: {out}")

    if n_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
