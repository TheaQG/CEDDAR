import os
import sys
from pathlib import Path
import zarr


def convert_zarr_to_zip(zarr_dir: str, zip_path: str):
    print(f"Converting: {zarr_dir} -> {zip_path}")

    p = Path(zarr_dir)
    print(f"[INFO] Exists: {p.exists()}  Is dir: {p.is_dir()}")
    if p.exists() and p.is_dir():
        try:
            print("[INFO] First entries in source dir:")
            for name in sorted(os.listdir(p))[:20]:
                print("   ", name)
        except Exception as e:
            print(f"[WARN] Could not list source dir: {e}")

    # Try the most direct open first
    try:
        src = zarr.open(zarr_dir, mode="r")
    except Exception as e1:
        print(f"[WARN] zarr.open failed: {e1}")
        try:
            src = zarr.open_group(store=zarr.storage.DirectoryStore(zarr_dir), mode="r")
        except Exception as e2:
            print(f"[ERROR] DirectoryStore open also failed: {e2}")
            raise

    store = zarr.storage.ZipStore(zip_path, mode="w")
    try:
        zarr.copy_store(src.store, store)
    finally:
        store.close()

    print(f"Done: {zip_path}")


if __name__ == "__main__":
    zarr_dir = sys.argv[1]
    zip_path = sys.argv[2]
    convert_zarr_to_zip(zarr_dir, zip_path)