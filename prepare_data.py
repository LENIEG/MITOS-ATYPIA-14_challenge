"""
Extracts all .tar.gz archives from the MITOS-ATYPIA-14 data folder into a
clean directory tree under data/extracted/.

Output structure:
  data/extracted/
    training/
      aperio/   A03/ A04/ ... (frames/, mitosis/, atypia/)
      hamamatsu/ H03/ H04/ ...
    testing/
      aperio/   A06/ A08/ ...
      hamamatsu/ H06/ H08/ ...

Run once:  python prepare_data.py
"""

import tarfile
from pathlib import Path

DATA_ROOT = Path(__file__).parent / "data"
EXTRACT_ROOT = DATA_ROOT / "extracted"

SOURCES = [
    ("mitos_atypia_2014_training_aperio",   "training/aperio"),
    ("mitos_atypia_2014_training_hamamatsu","training/hamamatsu"),
    ("mitos_atypia_2014_testing_aperio",    "testing/aperio"),
    ("mitos_atypia_2014_testing_hamamatsu", "testing/hamamatsu"),
]

def extract_all(force: bool = False) -> None:
    for src_folder, dest_rel in SOURCES:
        # archives live one level deeper (duplicate subfolder)
        archive_dir = DATA_ROOT / src_folder / src_folder
        dest_dir = EXTRACT_ROOT / dest_rel

        archives = sorted(archive_dir.glob("*.tar.gz"))
        if not archives:
            print(f"[WARN] No archives found in {archive_dir}")
            continue

        for archive in archives:
            slide_id = archive.stem.split(".")[0]          # e.g. "A03"
            out_dir = dest_dir / slide_id

            if out_dir.exists() and not force:
                print(f"[SKIP] {slide_id} already extracted -> {out_dir}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[EXTRACTING] {archive.name} -> {out_dir} ...", end=" ", flush=True)

            with tarfile.open(archive, "r:gz") as tf:
                # Strip the top-level folder (e.g. "A03/") from name AND linkname
                members = []
                for member in tf.getmembers():
                    parts = Path(member.name).parts
                    if len(parts) <= 1:
                        continue
                    member.name = str(Path(*parts[1:]))
                    # Hard-links store their target in linkname; strip it too
                    if member.linkname:
                        lparts = Path(member.linkname).parts
                        if len(lparts) > 1:
                            member.linkname = str(Path(*lparts[1:]))
                    members.append(member)
                tf.extractall(path=out_dir, members=members, filter="data")

            print("done")

    print("\nAll done. Extracted data is in:", EXTRACT_ROOT)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract MITOS-ATYPIA-14 archives.")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if output folder already exists.")
    args = parser.parse_args()
    extract_all(force=args.force)
