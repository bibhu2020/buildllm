import os
import sys
import torch
import tiktoken
# Load environment variables explicitly
from dotenv import load_dotenv
load_dotenv(override=True)

torch.manual_seed(123)

# When running `python src/gpt2/main.py` the package context isn't set,
# so add the repository `src` directory to `sys.path` to allow absolute imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.gpt2.core.dataloader import create_dataloader_v1
from src.llm.gpt2.core.attention import MultiHeadAttention
from src.llm.gpt2.core.model import GPTModel
from src.llm.gpt2.core.text_generator import generate_text_simple
from src.llm.gpt2.core.loss_calculation import calc_loss_loader
import torch.nn as nn

def main():
    DEBUG = {
        "print_dataset": False,
        "print_dataloader": False,
        "print_attention": False,
        "print_model_logits": False,
        "print_model_prediction": True,
        "upload_model_to_hf": False,
        "calculate_loss": False,
    }
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size inase of gpt2
        "context_length": 256,  # Context length is 1024 incase of gpt2
        "emb_dim": 768,          # Embedding dimension inase of gpt2
        "n_heads": 12,           # Number of attention heads (it decides the number of heads in the multi-head attention mechanism processing in parallel)
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    path_to_data = "./_data/the-verdict.txt"

    # load text file
    with open(path_to_data, "r", encoding="utf-8") as f:
        txt = f.read()

    # create a dataloader using the dataset helper
    # It returns you dataloader of tensors of shape (batch_size, context_length)
    # the tensor contains the tokenized input and target sequences
    # sanity check
    total_chars = len(txt)
    total_tokens = len(tokenizer.encode(txt))

    if total_tokens * 0.1 < GPT_CONFIG_124M["context_length"]:
        print("The dataset is too small to train on. Please use a larger dataset.")
        return


    # split train and test data
    split_idx = int(0.9 * len(txt))
    train_txt, test_txt = txt[:split_idx], txt[split_idx:]

    # create a dataloader using the dataset helper function
    train_dataloader = create_dataloader_v1(train_txt, 
        batch_size=2, 
        max_length=GPT_CONFIG_124M["context_length"], 
        stride=GPT_CONFIG_124M["context_length"]      # in gpt, stride is equal to max_length
    )  
    test_dataloader = create_dataloader_v1(test_txt, 
        batch_size=2, 
        max_length=GPT_CONFIG_124M["context_length"], 
        stride=GPT_CONFIG_124M["context_length"]
    )

    # DEBUG #1: Decode and print the tokenized input and target sequences
    if DEBUG["print_dataloader"]:
        tokenizer = tiktoken.get_encoding("gpt2")
        try:
            total_batches = len(train_dataloader)
        except Exception:
            total_batches = None

        for batch_idx, (input_batch, target_batch) in enumerate(train_dataloader, start=1):
            for seq_idx, seq in enumerate(input_batch):
                decoded = tokenizer.decode(seq.tolist())
                snippet = decoded[:25] + ("..." if len(decoded) > 25 else "")
                if total_batches:
                    print(f"Batch {batch_idx}/{total_batches} Seq {seq_idx} Input:", snippet)
                else:
                    print(f"Batch {batch_idx} Seq {seq_idx} Input:", snippet)
            for seq_idx, seq in enumerate(target_batch):
                decoded = tokenizer.decode(seq.tolist())
                snippet = decoded[:25] + ("..." if len(decoded) > 25 else "")
                if total_batches:
                    print(f"Batch {batch_idx}/{total_batches} Seq {seq_idx} Target:", snippet)
                else:
                    print(f"Batch {batch_idx} Seq {seq_idx} Target:", snippet)

    # create a MultiHeadAttention module
    attention = MultiHeadAttention(
        d_in=GPT_CONFIG_124M["emb_dim"], 
        d_out=GPT_CONFIG_124M["emb_dim"],
        context_length=GPT_CONFIG_124M["context_length"],
        num_heads=GPT_CONFIG_124M["n_heads"],
        dropout=GPT_CONFIG_124M["drop_rate"],
        qkv_bias=GPT_CONFIG_124M["qkv_bias"]
    )

    # embedding layer to convert token ids -> embeddings expected by attention
    embed = nn.Embedding(GPT_CONFIG_124M["vocab_size"], GPT_CONFIG_124M["emb_dim"]) 

    # DEBUG #2: Print the contextual vectors after attention
    if DEBUG["print_attention"]:
        for batch_idx, (input_batch, target_batch) in enumerate(train_dataloader, start=1):
            # input_batch: (B, L) of token ids (integers)
            print(f"Shape of input_batch: {input_batch.shape}")

            # convert to long and embed -> (B, L, emb_dim)
            input_batch = input_batch.long()
            x = embed(input_batch)
            print(f"After embedding, shape: {x.shape}")

            # pass the embeddings through the attention module
            # (B, L, emb_dim) -> (B, L, emb_dim)
            context_vecs = attention(x)
            print("final contextual verctors shape:", context_vecs.shape)
            print(f"\nfinal contextual verctors: {context_vecs}\n")
            break

    gpt2 = GPTModel(GPT_CONFIG_124M)

    # DEBUG #3: Print the model output (called logits)
    if DEBUG["print_model_logits"]:
        for batch_idx, (input_batch, target_batch) in enumerate(train_dataloader, start=1):
            # input_batch: (B, L) of token ids (integers)
            print(f"Shape of input_batch: {input_batch.shape}")

            logits = gpt2(input_batch)
            print(f"\nlogits: {logits}\n")
            print("logits shape:", logits.shape)
            break


    # DEBUG #4: print model predictions (model is not trained yet)
    # pass a few lines of text from the book expecting it to predict the next few tokens
    # but you will see random predictions as the model is not trained yet
    if DEBUG["print_model_prediction"]:
        tokenizer = tiktoken.get_encoding("gpt2")

        start_context = "life of the Riviera lends itself" # this a phrase from the book "The Verdict"

        encoded = tokenizer.encode(start_context)
        # print("encoded:", encoded)

        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        # print("encoded_tensor.shape:", encoded_tensor.shape)

        gpt2.eval() # disable dropout

        out = generate_text_simple(
            model=gpt2,
            idx=encoded_tensor,
            max_new_tokens=6,
            context_size=GPT_CONFIG_124M["context_length"]
        )

        # print("Output:", out)
        print("Output length:", len(out[0]))

        decoded_text = tokenizer.decode(out.squeeze(0).tolist())
        print(decoded_text)
        print("\nNote: **predicted output is not correct as the model is not trained yet.**")

    # DEBUG #5: Upload the model to Hugging Face
    if DEBUG["upload_model_to_hf"]:
        model_name = "gpt2-raw"  # Replace with your desired model name
        # build README for a raw (untrained) model with YAML front-matter
        readme_text = f"""---
    language: en
    license: mit
    tags:
      - gpt2
      - raw
      - untrained
    library_name: pytorch
    ---

    # {model_name}

    This repository contains a RAW / untrained checkpoint exported from the local `GPTModel` implementation in this repository.

    This is a GPT-2 style model implemented from scratch by the developer (not a Hugging Face pretrained release). It is an untrained initialization and can be used as a starting point for pre-training or fine-tuning on your own dataset.

    IMPORTANT: This is an untrained model checkpoint — DO NOT expect sensible outputs until the model is pre-trained or fine-tuned.

    Use cases:
    - store the architecture and checkpoint before training
    - pre-train on your dataset to obtain a usable language model
    - fine-tune after pre-training for specific tasks

    Model config (local):
    - vocab_size: {GPT_CONFIG_124M['vocab_size']}
    - context_length: {GPT_CONFIG_124M['context_length']}
    - emb_dim: {GPT_CONFIG_124M['emb_dim']}
    - n_layers: {GPT_CONFIG_124M['n_layers']}
    - n_heads: {GPT_CONFIG_124M['n_heads']}

    How to load the checkpoint (recommended for Hub users):

    1) Download the raw state_dict with `huggingface_hub.hf_hub_download` and load with PyTorch:

    ```python
    from huggingface_hub import hf_hub_download
    import torch

    # download the saved state_dict from the Hub
    path = hf_hub_download(repo_id="your-username/your-repo", filename="pytorch_model.bin")

    # you must recreate the same model architecture in Python (see `src/gpt2/core/model.py`)
    from src.gpt2.core.model import GPTModel
    cfg = {GPT_CONFIG_124M}
    model = GPTModel(cfg)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    ```

    2) Or download the whole repo snapshot (includes README, tokenizer when provided):

    ```python
    from huggingface_hub import snapshot_download
    import os, torch

    local_dir = snapshot_download(repo_id="your-username/your-repo")
    state_path = os.path.join(local_dir, "pytorch_model.bin")
    state = torch.load(state_path, map_location="cpu")
    # instantiate model as above and load state
    ```

    3) Loading tokenizer (if uploaded and compatible with `transformers`):

    ```python
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("your-username/your-repo")
    ```

    Note: If the Hub repo only contains a `state_dict` (not a full `transformers`-compatible checkpoint), users must have access to the model class (for example by cloning this repo or copying `src/gpt2/core/model.py`) in order to instantiate the architecture before calling `load_state_dict()` as shown above.

    If you want to upload to Hugging Face Hub, set the environment variable `HUGGINGFACE_HUB_TOKEN` or pass a token to the upload function.
    """

        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        try:
            repo_url, local_dir = gpt2.upload_model_to_hf(
                model_name=model_name,
                token=token,
                tokenizer_hf=None,
                private=False,
                readme_text=readme_text,
            )
            print(f"Uploaded model to {repo_url} (local files at: {local_dir})")
        except Exception as e:
            print("Failed to upload model to Hugging Face:", e)
    
            
    # DEBUG #6: Calculate and print the loss
    if DEBUG["calculate_loss"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpt2.to(device)
        torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader
        with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
            train_loss = calc_loss_loader(train_dataloader, gpt2, device)
            val_loss = calc_loss_loader(test_dataloader, gpt2, device)

        print("Training loss:", train_loss)
        print("Validation loss:", val_loss)


if __name__ == "__main__":
    main()