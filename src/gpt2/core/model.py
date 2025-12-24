import torch
import tiktoken
import torch.nn as nn
from .transformer import TransformerBlock
from .transformer import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # It stacks 12 transformer blocks together
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # print(f"Shape of the final normalization layer output: {self.final_norm.shape}\n")
        # print(f"the final normalization layer output: {self.final_norm}\n")
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # embedding + psotional embedding takes place here
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]

        # dropout takes place
        x = self.drop_emb(x)

        # transformer blocks are invoked
        x = self.trf_blocks(x)

        # final normalization layer
        x = self.final_norm(x)

        # converts the output to logits which is probability distribution on the vocalbulary (50,257)
        logits = self.out_head(x)

        return logits
    
    def upload_model_to_hf(self, model_name="my_model", token=None, tokenizer_hf=None, private=False, meta_cfg=None, readme_text: str = None, max_retries: int = 3):
        """Save model artifacts under `models/<model_name>` and upload to HF.

        This method creates a local folder `models/<model_name>` (nested for
        namespace like "user/repo"), writes model artifacts and a `README.md`
        (from `readme_text`), and uploads all files from that folder to the
        Hugging Face Hub.

        Args:
            model_name (str): repository name on HF (e.g. "user/repo" or "repo").
            token (str, optional): HF token; if not provided will use env var or `meta_cfg`.
            tokenizer_hf (PreTrainedTokenizer, optional): a Hugging Face tokenizer
                instance with `save_pretrained()` to include tokenizer files.
            private (bool, optional): create the HF repo as private when True.
            meta_cfg (dict, optional): optional metadata/config dict to read fallback values from.
            readme_text (str, optional): text to write into `README.md` in the model folder.

        Returns:
            tuple: (repo_url: str, local_dir: str)
        """
        import os
        import json

        # allow explicit args to override entries in meta_cfg
        if isinstance(meta_cfg, dict):
            if token is None:
                token = meta_cfg.get("hf_token")
            if tokenizer_hf is None:
                tokenizer_hf = meta_cfg.get("tokenizer_hf")
            if not private:
                private = meta_cfg.get("hf_private", private)

        # build local directory under `models/` (support namespaced repo)
        parts = model_name.split("/") if "/" in model_name else [model_name]
        local_dir = os.path.join("models", *parts)
        os.makedirs(local_dir, exist_ok=True)

        # save state_dict and full model into local_dir
        state_path = os.path.join(local_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), state_path)

        safe_name = model_name.replace("/", "_")
        full_path = os.path.join(local_dir, f"{safe_name}.pth")
        torch.save(self, full_path)

        # minimal config metadata
        cfg = {
            "model_type": "custom",
            "hidden_size": getattr(self.out_head, "in_features", None),
            "vocab_size": getattr(self.out_head, "out_features", None),
        }
        config_path = os.path.join(local_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f)

        # save tokenizer if provided
        if tokenizer_hf is not None:
            tok_dir = os.path.join(local_dir, "tokenizer")
            os.makedirs(tok_dir, exist_ok=True)
            tokenizer_hf.save_pretrained(tok_dir)

        # write README.md
        readme_path = os.path.join(local_dir, "README.md")
        if readme_text is None:
            readme_text = f"Model {model_name} saved from GPTModel instance."
        with open(readme_path, "w") as f:
            f.write(readme_text)

        # upload using huggingface_hub
        try:
            from huggingface_hub import HfApi
        except Exception as e:
            raise RuntimeError("Please install huggingface_hub to upload to the Hub: `pip install huggingface_hub`\n" + str(e))

        # load .env from repository root if present so users can keep HF_TOKEN there
        repo_root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
        if os.path.exists(repo_root_env):
            try:
                # prefer python-dotenv if installed
                from dotenv import load_dotenv

                load_dotenv(repo_root_env, override=False)
            except Exception:
                # simple fallback parser (KEY=VALUE, ignore comments)
                try:
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
                except Exception:
                    pass

        # always read token from HF_TOKEN environment variable
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise RuntimeError(
                "No Hugging Face token found in HF_TOKEN environment variable. "
                "Set HF_TOKEN (for example in a .env at the repo root) before calling upload_model_to_hf()."
            )

        api = HfApi(token=token)
        # if model_name is not namespaced, resolve the authenticated username
        repo_id = model_name
        try:
            if "/" not in model_name:
                who = api.whoami()
                username = who.get("name") or who.get("login") or who.get("user") or who.get("id")
                if not username:
                    raise RuntimeError("Unable to determine HF username from token; please provide a namespaced `model_name` like 'username/repo'.")
                repo_id = f"{username}/{model_name}"
        except Exception as e:
            raise RuntimeError(f"Failed to resolve HF username from token: {e}")

        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
        except Exception as e:
            raise RuntimeError(f"Failed to create or access repo '{repo_id}': {e}")

        def _upload(local_path, repo_path):
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to upload '{local_path}' to '{repo_id}/{repo_path}': {e}")

        # prefer `upload_folder` which is optimized for folders and will handle
        # chunking/retries; fall back to per-file upload on error
        try:
            from huggingface_hub import upload_folder

            attempt = 0
            while True:
                try:
                    upload_folder(
                        folder_path=local_dir,
                        repo_id=repo_id,
                        path_in_repo="",
                        token=token,
                        repo_type="model",
                    )
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= max_retries:
                        raise
                    wait = 2 ** attempt
                    print(f"upload_folder failed (attempt {attempt}/{max_retries}), retrying in {wait}s: {e}")
                    import time

                    time.sleep(wait)
        except Exception:
            # fallback to individual file uploads (preserve relative paths)
            for root, _, files in os.walk(local_dir):
                for fname in files:
                    local = os.path.join(root, fname)
                    rel = os.path.relpath(local, local_dir)
                    # use forward slashes in repo path
                    repo_path = rel.replace(os.path.sep, "/")
                    _upload(local, repo_path)

        repo_url = f"https://huggingface.co/{repo_id}"
        return repo_url, local_dir


__all__ = ["GPTModel"]