"""Upload the app to the Hugging Face Space (Gradio SDK) that hosts the website.

Run from the repo root after `hf auth login` (needs a token with Write access):
    python scripts/upload_space.py NotALemon/gated-restoration

Why a script instead of `hf upload ... --exclude "x/*"`: on Windows the hf command expands the
`*` in each pattern into file names, which breaks the command. Here the patterns go straight
to the library.

What it uploads: app.py, app/, static/, scripts/ and deploy/space_requirements.txt (as the
Space's requirements.txt). The Space keeps its own README.md (it holds the Space settings);
weights are not uploaded (app.py downloads them when the Space starts).
"""
import sys
from pathlib import Path

from huggingface_hub import upload_file, upload_folder

ROOT = Path(__file__).resolve().parents[1]

IGNORE = [
    "hf_weights/*", ".git/*", ".vibe-wise/*", "handoff/*", "tests/*", "deploy/*",
    "README.md", "requirements.txt", "Dockerfile", ".dockerignore", "*__pycache__*",
    "TEAMMATE_CONTEXT.md", ".gitignore", "pytest.ini",
]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/upload_space.py <hf-username>/<space-name>")
    space = sys.argv[1]
    upload_folder(repo_id=space, repo_type="space", folder_path=ROOT, ignore_patterns=IGNORE,
                  commit_message="Upload app")
    upload_file(repo_id=space, repo_type="space", path_or_fileobj=ROOT / "deploy" / "space_requirements.txt",
                path_in_repo="requirements.txt", commit_message="Space requirements (CPU PyTorch)")
    print(f"Uploaded. Watch the build: https://huggingface.co/spaces/{space}?logs=build")


if __name__ == "__main__":
    main()
