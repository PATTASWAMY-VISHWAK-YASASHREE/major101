"""
Phase 0: Download C-BRATS (Synapse) and IBSR (NITRC).
Set environment variables SYNAPSE_USER and SYNAPSE_PASSWORD before running.
"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
C_BRATS_DIR = DATA_DIR / "raw" / "c-brats"
IBSR_DIR = DATA_DIR / "raw" / "ibsr"


def download_cbrats():
    """Download C-BRATS from Synapse."""
    user = os.environ.get("SYNAPSE_USER")
    pwd = os.environ.get("SYNAPSE_PASSWORD")
    if not user or not pwd:
        print("[c-brats] Skipping: set SYNAPSE_USER and SYNAPSE_PASSWORD env vars.")
        print("[c-brats] Or download manually: https://www.synapse.org/Synapse:syn2582906")
        return

    try:
        import synapseclient
    except ImportError:
        print("[c-brats] pip install synapseclient required")
        return

    C_BRATS_DIR.mkdir(parents=True, exist_ok=True)
    syn = synapseclient.Synapse()
    syn.login(user, pwd)

    synapse_id = "syn2582906"
    print(f"[c-brats] Downloading {synapse_id} to {C_BRATS_DIR} ...")
    entity = syn.get(synapse_id, downloadLocation=str(C_BRATS_DIR))

    if hasattr(entity, "children"):
        for child in syn.getChildren(entity):
            syn.get(str(child["id"]), downloadLocation=str(C_BRATS_DIR))
            print(f"  Downloaded: {child['name']}")

    print(f"[c-brats] Done. Check {C_BRATS_DIR}")


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
    for name, d in [("c-brats", C_BRATS_DIR), ("ibsr", IBSR_DIR)]:
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
        download_cbrats()
        download_ibsr()
