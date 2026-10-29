"""Download the three model checkpoints from Hugging Face into hf_weights/ and verify them.

Run from the repo root:  python scripts/download_weights.py
Safe to re-run: files that are already present with the right hash are not downloaded again.

The revision is pinned so everyone (laptops, Colab) gets byte-identical files; the SHA-256 values
are of those files. hf_weights/ is git-ignored. The ".zip" names are historical: the files are
plain PyTorch checkpoints and must NOT be unzipped (app/models/*.py load them as they are).
"""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ananth2007/gated-restoration-sidd"
REVISION = "fb03fd2480673018dc2b40905ce3cd1dc6aa3b19"
DEST = Path(__file__).resolve().parents[1] / "hf_weights"

# filename -> SHA-256 of the file at REVISION
FILES = {
    "nafnet.pth.zip": "53e58656df4f04b3deba61a332df3b51de7eaf6622d7acbad9df43e7182ae799",
    "irsde.pth.zip": "5e84f428b65bc34568030d56e15de1dd047e9581bee9bd8ff853cf15caff50f6",
    "umap_generator.pt.zip": "cf092851ad08f6649679ade414669e3c6bfb8fb21d0497ac95b074ee8869099d",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DEST.mkdir(exist_ok=True)
    failed = []
    for name, expected in FILES.items():
        path = DEST / name
        if path.exists() and sha256(path) == expected:
            print(f"ok (already present)  {name}")
            continue
        print(f"downloading           {name}")
        hf_hub_download(REPO_ID, name, revision=REVISION, local_dir=DEST)
        if sha256(path) == expected:
            print(f"ok                    {name}")
        else:
            print(f"HASH MISMATCH         {name}: delete it and re-run")
            failed.append(name)
    if failed:
        sys.exit(1)
    print(f"All weights are in {DEST}")


if __name__ == "__main__":
    main()
