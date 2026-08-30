#!/usr/bin/env python3
"""PROTEAS downloader — stage 1 (verify set) or full pull.

Pure Python 3.9+ stdlib. No pip installs. Runs on any machine.
Downloads from Zenodo record 17253793 with md5 verification + resume.

Usage:
  python download_proteas.py --out D:\\proteas_raw            # stage 1: xlsx + P01-P03
  python download_proteas.py --out D:\\proteas_raw --all      # full dataset
  python download_proteas.py --out D:\\proteas_raw --patients P05 P06   # specific
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
import zipfile

RECORD = "17253793"
API_FILES = f"https://zenodo.org/api/records/{RECORD}/files"
PAGE_FILES = f"https://zenodo.org/records/{RECORD}/files"

# md5s scraped from the record page (2026-08-31). The API is the source of truth
# at run time; this table is a fallback and cross-check.
KNOWN_MD5 = {
    "P01.zip": "6f8c6ba98a275d4a7864f4d3cc5f26a2",
    "P02.zip": "b998b27c764fb68f09503100f4e32032",
    "P03.zip": "b3b8672949372c70abb5ebc0409da519",
}

STAGE1_PATIENTS = ["P01", "P02", "P03"]


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_json(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "proteas-downloader/1.0 (research)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def get_file_list():
    """Return [(name, size, md5, download_url), ...] from the Zenodo API."""
    data = fetch_json(API_FILES)
    # API returns {"entries": [...]} on new Zenodo; older returns a list
    entries = data.get("entries") if isinstance(data, dict) else data
    if entries is None and isinstance(data, dict):
        entries = data.get("files", [])
    out = []
    for e in entries:
        name = e.get("key") or e.get("filename")
        if not name:
            continue
        out.append({
            "name": name,
            "size": e.get("size"),
            "md5": (e.get("checksum") or "").replace("md5:", ""),
            "url": (e.get("links") or {}).get("self") or e.get("download_url")
                   or f"https://zenodo.org/records/{RECORD}/files/{urllib.parse.quote(name)}?download=1",
        })
    if not out:
        raise RuntimeError("Zenodo API returned no files — record layout changed; open the record page manually")
    return out


def download_file(entry, out_dir, tries=3):
    """Download one file with resume; return 'ok' | 'skip' (already verified) | 'fail'."""
    dest = os.path.join(out_dir, entry["name"])
    part = dest + ".part"
    expect_md5 = entry["md5"] or KNOWN_MD5.get(entry["name"])

    if os.path.exists(dest):
        if expect_md5 and md5sum(dest) == expect_md5:
            return "skip"
        os.remove(dest)  # corrupt or unverifiable -> redo

    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(entry["url"], headers={"User-Agent": "proteas-downloader/1.0 (research)"})
            mode = "ab" if os.path.exists(part) else "wb"
            with urllib.request.urlopen(req, timeout=120) as r, open(part, mode) as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = os.path.getsize(part) if mode == "ab" else 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total and done % (50 << 20) < (1 << 20):  # ~every 50MB
                        print(f"    {entry['name']}: {done/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
            if expect_md5:
                got = md5sum(part)
                if got != expect_md5:
                    os.remove(part)
                    raise RuntimeError(f"md5 mismatch (got {got})")
            os.replace(part, dest)
            return "ok"
        except Exception as e:  # noqa: BLE001
            print(f"    attempt {attempt}/{tries} failed: {e}", flush=True)
            time.sleep(3 * attempt)
    if os.path.exists(part):
        os.remove(part)
    return "fail"


def main():
    import urllib.parse  # noqa: F401  (used inside get_file_list via fully-qualified name fix below)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory, e.g. D:\\proteas_raw")
    ap.add_argument("--all", action="store_true", help="download every file in the record")
    ap.add_argument("--patients", nargs="*", help="specific patient zips, e.g. P05 P06")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Fetching file list from Zenodo API ...")
    files = get_file_list()
    names = [f["name"] for f in files]
    print(f"Record has {len(files)} files.")

    if args.all:
        targets = files
    elif args.patients:
        targets = []
        for p in args.patients:
            pfx = p if p.endswith(".zip") else p + ".zip"
            match = [f for f in files if f["name"] == pfx or f["name"].startswith(pfx[:-4] + "a") or f["name"].startswith(pfx[:-4] + "b")]
            targets += match
    else:
        # Stage 1: all non-zip small files (xlsx) + the three stage-1 patient zips
        targets = [f for f in files if not f["name"].endswith(".zip")]
        for p in STAGE1_PATIENTS:
            targets += [f for f in files if f["name"].startswith(p)]

    seen = set()
    targets = [t for t in targets if not (t["name"] in seen or seen.add(t["name"]))]
    total_mb = sum((t["size"] or 0) for t in targets) / 1e6
    print(f"Downloading {len(targets)} file(s), ~{total_mb:.0f} MB total -> {args.out}\n")

    results = {}
    for t in targets:
        print(f"[{t['name']}] ({(t['size'] or 0)/1e6:.1f} MB)")
        results[t["name"]] = download_file(t, args.out)
        print(f"  -> {results[t['name']]}")

    ok = [n for n, r in results.items() if r != "fail"]
    fail = [n for n, r in results.items() if r == "fail"]
    print(f"\nDone. ok/skip: {len(ok)}, failed: {len(fail)}")
    if fail:
        print("FAILED (re-run the same command — verified files are skipped):")
        for n in fail:
            print("  -", n)
        sys.exit(1)
    print("All verified. Next: python audit_proteas.py --src " + args.out)


if __name__ == "__main__":
    main()
