"""
Phase 0: Download BraTS 2024 GLI (Synapse) and IBSR (NITRC).
Set environment variables SYNAPSE_USER and SYNAPSE_PASSWORD before running.
"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
BRATS_DIR = DATA_DIR / "raw" / "brats2024"
IBSR_DIR = DATA_DIR / "raw" / "ibsr"

# Synapse IDs for BraTS 2024 GLI
SYN_IDS = {
    "training": "syn60086071",   # BraTS2024-BraTS-GLI-TrainingData.zip (34.89 GB)
    "validation": "syn61455507",  # BraTS2024-BraTS-GLI-ValidationData.zip (4.99 GB)
}


def download_brats():
    """Download BraTS 2024 GLI from Synapse."""
    user = os.environ.get("SYNAPSE_USER")
    pwd = os.environ.get("SYNAPSE_PASSWORD")
    if not user or not pwd:
        print("[brats] Skipping: set SYNAPSE_USER and SYNAPSE_PASSWORD env vars.")
        print("[brats] Or download manually: https://www.synapse.org/Synapse:syn2582906")
        return

    try:
        import synapseclient
    except ImportError:
        print("[brats] pip install synapseclient required")
        return

    BRATS_DIR.mkdir(parents=True, exist_ok=True)
    syn = synapseclient.Synapse()
    syn.login(user, pwd)

    for name, syn_id in SYN_IDS.items():
        dest = BRATS_DIR / name
        dest.mkdir(exist_ok=True)
        print(f"[brats] Downloading {name} ({syn_id}) to {dest} ...")
        syn.get(syn_id, downloadLocation=str(dest))
        print(f"  Done: {name}")

    print(f"[brats] Done. Check {BRATS_DIR}")


def download_ibsr():
    """Download IBSR from NITRC."""
    IBSR_DIR.mkdir(parents=True, exist_ok=True)
    print("[ibsr] IBSR download requires manual steps:")
    print("  1. Go to: https://www.nitrc.org/projects/ibsr/")
    print("  2. Download IBSR cases (T1 + CT per case)")
    print("  3. Place in: data/raw/ibsr/IBSR_001/, IBSR_002/, ...")
    print("  Alternative: https://figshare.com/ndownloader/files/3828419")


def verify_downloads():
    """Check that data directories have files."""
    for name, d in [("brats", BRATS_DIR), ("ibsr", IBSR_DIR)]:
        if not d.exists():
            print(f"[verify] {name}: directory not found")
            continue
        count = len(list(d.rglob("*.nii.gz")))
        if count == 0:
            print(f"[verify] {name}: no .nii.gz files found")
        else:
            total_bytes = sum(f.stat().st_size for f in d.rglob("*.nii.gz"))
            print(f"[verify] {name}: {count} files, {total_bytes/1e9:.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_downloads()
    else:
        download_brats()
        download_ibsr()
