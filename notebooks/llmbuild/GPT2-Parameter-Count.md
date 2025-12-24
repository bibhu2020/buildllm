# Parameter Count in GPT2

## Embedding Parameters

Token Embeddings = (vocab_size x hidden_dim) = 50257 x 768

Positional Embeddings = (context_size x hidden_dim) = 1024 x 768

Total = Token Embeddings + Positional Embeddings = 38.4M

## Transformer

### Multi-head Attention

Query + Key + Value Weights = 3 x 768 x 768 = 1.77M
Output Head = 768 x 768 = 0.59M
Total = 2.36M

### Feed Forward Network

768 x (4 x 768 ) + 768 x (4 x 768 ) = 4.72 M

We have 12 Transformer Block

So, total from Transformer Block = 12 x (2.36M + 4.72M) = 85.2M


## Final Layer (logits)

768 x 50257 = 38.4M

## Total

= 38.4M + 85.2M + 38.4M = 162M