"""Upload runtime data to the Hugging Face dataset repo (deploy prerequisite).

Runtime data (ChromaDB index, the graph pickle, trials.jsonl) is intentionally kept
OUT of git — it is large and pipeline-generated (see .gitignore). The HF Space fetches
it at startup via app._ensure_data(). This script is the other half of that contract:
it pushes the current local data to the dataset repo so the Space has fresh data to pull.

Run it BEFORE `git push hf main` on every deploy — otherwise the Space serves whatever
data the dataset repo last had (or, if a file was never uploaded, none — which silently
breaks features like trial search).

What it uploads (mirrors _ensure_data exactly):
  data/chroma/            → chroma/            (the ChromaDB store)
  data/graph/als_graph.pkl → graph/als_graph.pkl
  data/trials/trials.jsonl → trials/trials.jsonl

Usage:
  uv run python scripts/upload_data.py            # upload everything
  uv run python scripts/upload_data.py --dry-run  # list what would upload, no writes
  uv run python scripts/upload_data.py --only trials   # one target (chroma|graph|trials)

Auth: needs a Hugging Face token with write access to the dataset repo. Provide it via
`huggingface-cli login`, or the HF_TOKEN / HUGGING_FACE_HUB_TOKEN env var.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/upload_data.py` (repo root on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHROMA_DIR, GRAPH_PICKLE_PATH, HF_DATASET_REPO, TRIALS_PATH  # noqa: E402


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}GB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload runtime data to the HF dataset repo.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would upload; write nothing.")
    parser.add_argument("--only", choices=["chroma", "graph", "trials"],
                        help="Upload just one target (default: all).")
    parser.add_argument("--repo", default=HF_DATASET_REPO, help="Override the dataset repo id.")
    args = parser.parse_args()

    # (key, kind, local_path, path_in_repo)
    targets = [
        ("chroma", "folder", CHROMA_DIR, "chroma"),
        ("graph", "file", GRAPH_PICKLE_PATH, "graph/als_graph.pkl"),
        ("trials", "file", TRIALS_PATH, "trials/trials.jsonl"),
    ]
    if args.only:
        targets = [t for t in targets if t[0] == args.only]

    # Verify every selected target exists locally before touching the network.
    missing = [str(p) for _, _, p, _ in targets if not p.exists()]
    if missing:
        print("ERROR — local data missing (run the offline pipeline first):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    print(f"Dataset repo: {args.repo}")
    for key, kind, local, in_repo in targets:
        size = _dir_size(local) if kind == "folder" else local.stat().st_size
        print(f"  {key:7} {kind:6} {local}  ({_human_size(size)})  →  {in_repo}")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        who = api.whoami().get("name")
    except Exception:
        print("ERROR — not authenticated. Run `huggingface-cli login` or set HF_TOKEN.", file=sys.stderr)
        return 1
    print(f"Authenticated as: {who}\n")

    for key, kind, local, in_repo in targets:
        print(f"Uploading {key} ...")
        if kind == "folder":
            api.upload_folder(
                folder_path=str(local), path_in_repo=in_repo,
                repo_id=args.repo, repo_type="dataset",
                commit_message=f"Update {in_repo} (deploy)",
            )
        else:
            api.upload_file(
                path_or_fileobj=str(local), path_in_repo=in_repo,
                repo_id=args.repo, repo_type="dataset",
                commit_message=f"Update {in_repo} (deploy)",
            )
        print(f"  done: {in_repo}")

    print("\nAll runtime data uploaded. Now deploy the code: git push hf main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
