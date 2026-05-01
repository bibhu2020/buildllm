import pandas as pd
from datasets import Dataset

# Load raw text
with open("../data/ramayana.txt", "r", encoding="utf-8") as f:
    ramayana_text = f.read()
with open("../data/mahabharata.txt", "r", encoding="utf-8") as f:
    mahabharata_text = f.read()

# Optional: split into verses or paragraphs
def split_text(text, chunk_size=512):
    lines = text.split("\n")
    chunks = []
    current_chunk = ""
    for line in lines:
        current_chunk += line.strip() + " "
        if len(current_chunk.split()) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

data = []
for chunk in split_text(ramayana_text) + split_text(mahabharata_text):
    data.append({"text": chunk})

df = pd.DataFrame(data)
df.to_csv("../data/processed_dataset.csv", index=False)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset.save_to_disk("../data/hf_dataset")
