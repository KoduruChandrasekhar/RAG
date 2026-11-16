"""Publish FilingScope to a Hugging Face Space.

No GitHub involved: this uploads straight from this folder to the Space's own
repository.

    huggingface-cli login            # once, needs a token with 'write'
    python deploy/push_to_space.py --space <your-username>/filingscope

What gets uploaded:
    src/app.py, src/paths.py,
    src/query_engine.py                 the app and the retrieval engine
    data/parsed/                        ~3 MB, needed for the BM25 index
    data/embeddings/                    ~40 MB, the ChromaDB vector store
    README.md, requirements.txt         Space config and dependencies

What does not:
    data/raw/   236 MB of original filings, only needed to rebuild the index
    tests/, the ingest/parse/eval scripts, and .env

Answer generation is optional. To turn it on, add GEMINI_API_KEY as a Space
secret (Settings -> Variables and secrets). Without it the Space still
retrieves and shows passages, it just does not write an answer -- which is
often what you want for a public demo, since the Gemini free tier allows only
20 generations per day.
"""
import argparse
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
ROOT = os.path.dirname(SRC)
DEPLOY = os.path.join(ROOT, "deploy")


def build_parser():
    parser = argparse.ArgumentParser(description="Publish FilingScope to a HF Space.")
    parser.add_argument("--space", required=True, metavar="USER/NAME",
                        help="target Space, e.g. yourname/filingscope")
    parser.add_argument("--private", action="store_true",
                        help="create the Space as private")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be uploaded and exit")
    return parser


def planned_uploads():
    """(local path, path inside the Space) for everything that ships."""
    items = [
        (os.path.join(DEPLOY, "README_SPACE.md"), "README.md"),
        (os.path.join(DEPLOY, "requirements_space.txt"), "requirements.txt"),
        # src/ must stay a subfolder: paths.py derives the project root from
        # its own location, so flattening would point data/ outside the Space.
        (os.path.join(SRC, "app.py"), "src/app.py"),
        (os.path.join(SRC, "paths.py"), "src/paths.py"),
        (os.path.join(SRC, "query_engine.py"), "src/query_engine.py"),
    ]
    folders = [
        (os.path.join(ROOT, "data", "parsed"), "data/parsed"),
        (os.path.join(ROOT, "data", "embeddings"), "data/embeddings"),
    ]
    return items, folders


def human(n):
    return f"{n / 1e6:.1f} MB"


def folder_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def main():
    args = build_parser().parse_args()
    items, folders = planned_uploads()

    missing = [p for p, _ in items if not os.path.exists(p)]
    missing += [p for p, _ in folders if not os.path.isdir(p)]
    if missing:
        print("Missing, so the Space would be broken:")
        for path in missing:
            print(f"  {path}")
        print("\nBuild the index first:  python src/build_vector_db.py")
        return 1

    total = sum(os.path.getsize(p) for p, _ in items)
    total += sum(folder_size(p) for p, _ in folders)

    print(f"Space:  {args.space}")
    print(f"Upload: {human(total)}\n")
    for path, dest in items:
        print(f"  {dest:<24} {human(os.path.getsize(path))}")
    for path, dest in folders:
        print(f"  {dest + '/':<24} {human(folder_size(path))}")

    if args.dry_run:
        print("\nDry run, nothing uploaded.")
        return 0

    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        print("\nhuggingface_hub is not installed:  pip install huggingface_hub")
        return 1

    try:
        user = whoami()
    except Exception:
        print("\nNot logged in. Run:  huggingface-cli login")
        print("Create a token with 'write' access at "
              "https://huggingface.co/settings/tokens")
        return 1

    print(f"\nLogged in as {user.get('name')}.")

    api = HfApi()
    api.create_repo(repo_id=args.space, repo_type="space", space_sdk="streamlit",
                    private=args.private, exist_ok=True)

    for path, dest in items:
        print(f"  uploading {dest}")
        api.upload_file(path_or_fileobj=path, path_in_repo=dest,
                        repo_id=args.space, repo_type="space")

    for path, dest in folders:
        print(f"  uploading {dest}/ ({human(folder_size(path))})")
        api.upload_folder(folder_path=path, path_in_repo=dest,
                          repo_id=args.space, repo_type="space")

    url = f"https://huggingface.co/spaces/{args.space}"
    print(f"\nDone. The Space builds in a few minutes:\n  {url}")
    print("\nTo enable written answers, add GEMINI_API_KEY under "
          "Settings -> Variables and secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
