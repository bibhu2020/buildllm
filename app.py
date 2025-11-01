# ===============================================
# Imports
# ===============================================
import importlib
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# ===============================================
# Tokenizer Initialization
# ===============================================
tokenizer = tiktoken.get_encoding("gpt2")

# ===============================================
# Dataset Definition
# ===============================================
class GPTDatasetV1(Dataset):
    """
    Custom Dataset for GPT-like language modeling tasks.
    Uses a sliding window to create input-target token sequences.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Tokenized text must be longer than max_length."

        # Create overlapping sequences with a sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# ===============================================
# Dataloader Creation
# ===============================================
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, 
                         shuffle=True, drop_last=True, num_workers=0):
    """
    Creates a PyTorch DataLoader from raw text.
    """
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader, dataset

# ===============================================
# Embedding Layers
# ===============================================
def create_embedding_layers(vocab_size, max_length, embedding_dim=128):
    """
    Create token and positional embedding layers.
    """
    token_embedding_layer = torch.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim
    )

    positional_embedding_layer = torch.nn.Embedding(
        num_embeddings=max_length,
        embedding_dim=embedding_dim
    )

    return token_embedding_layer, positional_embedding_layer

# ===============================================
# Example Usage
# ===============================================
if __name__ == "__main__":
    # ----------------------------
    # Load raw text
    # ----------------------------
    with open("./_data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # ----------------------------
    # Parameters
    # ----------------------------
    batch_size = 2
    max_length = 4
    stride = 1

    # ----------------------------
    # Create dataloader
    # ----------------------------
    dataloader, dataset = create_dataloader_v1(
        raw_text, batch_size=batch_size, max_length=max_length, stride=stride, shuffle=False
    )

    # ----------------------------
    # Inspect first two batches
    # ----------------------------
    data_iter = iter(dataloader)
    for batch_num in range(2):
        input_ids, target_ids = next(data_iter)
        print(f"Batch {batch_num + 1}:")
        for i in range(input_ids.shape[0]):
            print("Decoded Input :", tokenizer.decode(input_ids[i].tolist()))
            print("Decoded Target:", tokenizer.decode(target_ids[i].tolist()))
        print("\n")

    # ----------------------------
    # Create embeddings
    # ----------------------------
    vocab_size = tokenizer.n_vocab
    token_emb, pos_emb = create_embedding_layers(vocab_size, max_length)

    # Example: combine token and positional embeddings for a batch
    input_embeddings = token_emb(input_ids) + pos_emb(torch.arange(max_length))
    print("Input Embeddings Shape:", input_embeddings.shape)

    # ===============================================
    # Generate Complete Encoded Dataset for GPT
    # ===============================================
    all_input_ids = torch.stack(dataset.input_ids)   # Shape: [num_sequences, max_length]
    all_target_ids = torch.stack(dataset.target_ids) # Shape: [num_sequences, max_length]

    print("Complete Dataset Shapes:")
    print("Input IDs:", all_input_ids.shape)
    print("Target IDs:", all_target_ids.shape)

    # Optional: Save for training
    torch.save({
        "input_ids": all_input_ids,
        "target_ids": all_target_ids
    }, "./outputs/gpt_training_data.pt")
    
    print("Complete encoded dataset saved to 'gpt_training_data.pt'.")
