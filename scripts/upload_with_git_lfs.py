#!/usr/bin/env python3
"""Upload a local models/<model_name> folder to Hugging Face using git + git-lfs.

Usage:
  python scripts/upload_with_git_lfs.py --model bpmgpt2-raw [--private]

This script:
- reads HF_TOKEN from environment (loads .env if present)
- creates the HF repo if missing
- clones the repo into a temp dir
- copies files from `models/<model_name>` into the clone
- runs `git lfs track` for common binary patterns
- commits and pushes using `huggingface_hub.Repository`

Requires: huggingface_hub, git, git-lfs installed and on PATH.
"""
import os
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path

def load_env_from_repo_root():
    repo_root_env = Path(__file__).resolve().parents[2] / ".env"
    if repo_root_env.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(str(repo_root_env), override=False)
        except Exception:
            with open(repo_root_env, "r") as ef:
                for line in ef:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)


def ensure_git_lfs():
    try:
        subprocess.run(["git", "lfs", "install"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise RuntimeError("git-lfs not available or failed to install: " + str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model name (repo name or repo slug). If not namespaced, your username will be prefixed")
    parser.add_argument("--private", action="store_true", help="create a private repo")
    parser.add_argument("--source", default=None, help="path to local model folder (defaults to models/<model>)")
    args = parser.parse_args()

    load_env_from_repo_root()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set; set it in .env or export it before running")

    try:
        from huggingface_hub import HfApi, Repository
    except Exception as e:
        raise RuntimeError("Please install huggingface_hub: pip install huggingface_hub")

    api = HfApi(token=token)
    # resolve username if needed
    repo_slug = args.model
    if "/" not in repo_slug:
        who = api.whoami()
        username = who.get("name") or who.get("login") or who.get("id")
        if not username:
            raise RuntimeError("Could not determine HF username from token; provide namespaced model name user/repo")
        repo_id = f"{username}/{repo_slug}"
    else:
        repo_id = repo_slug

    # create repo if needed
    api.create_repo(repo_id=repo_id, exist_ok=True, private=args.private)

    # source folder
    source = Path(args.source) if args.source else Path("models") / repo_slug.split("/")[-1]
    if not source.exists():
        raise RuntimeError(f"Source folder not found: {source}")

    ensure_git_lfs()

    tmpdir = Path(tempfile.mkdtemp(prefix="hf_upload_"))
    try:
        # clone the repo into temp dir
        repo = Repository(local_dir=str(tmpdir), clone_from=repo_id, use_auth_token=token)

        # run git lfs track for common binary patterns
        subprocess.run(["git", "lfs", "track", "*.bin", "*.pth", "*.pt"], check=False, cwd=str(tmpdir))

        # copy files from source into cloned repo
        for item in source.rglob("*"):
            rel = item.relative_to(source)
            dest = tmpdir / rel
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)

        # add, commit and push
        repo.push_to_hub(commit_message=f"Upload model {repo_id}")
        print(f"Uploaded {source} to https://huggingface.co/{repo_id}")
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
