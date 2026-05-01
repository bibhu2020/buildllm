import os
import sys
import torch
import tiktoken

# When running `python src/gpt2/main.py` the package context isn't set,
# so add the repository `src` directory to `sys.path` to allow absolute imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.gpt2.core.dataloader import create_dataloader_v1
from src.llm.gpt2.core.attention import MultiHeadAttention
from src.llm.gpt2.core.model import GPTModel
from src.llm.gpt2.core.text_generator import generate_text_simple
import torch.nn as nn

tokenizer = tiktoken.get_encoding("gpt2")

def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size inase of gpt2
        "context_length": 256,  # Context length is 1024 incase of gpt2
        "emb_dim": 768,          # Embedding dimension inase of gpt2
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    path_to_data = "./_data/the-verdict.txt"

    # load text file
    with open(path_to_data, "r", encoding="utf-8") as f:
        txt = f.read()

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
    
     
    # create a GPT model instance   
    model = GPTModel(GPT_CONFIG_124M)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):  # Number of epochs
        model.train()
        for batch in train_dataloader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, GPT_CONFIG_124M["vocab_size"]), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    ###################################################################
    # Generate text using the trained model
    start_context = "life of the Riviera lends itself" # this a phrase from the book "The Verdict"

    encoded = tokenizer.encode(start_context)
    # print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval() # disable dropout

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    # print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


if __name__ == "__main__":
    main()