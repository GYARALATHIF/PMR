from transformers import AutoTokenizer, AutoModel
import torch

# Load ClinicalBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Sample clinical note
note = """The patient presented with shortness of breath, elevated WBC count, and fever. Treated with IV antibiotics."""

# Tokenize and encode
tokens = tokenizer(note, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Get embeddings (last hidden state)
with torch.no_grad():
    outputs = model(**tokens)
    last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

# Average across tokens to get 768-dim embedding
embedding = last_hidden_state.mean(dim=1).squeeze().numpy()

print(" Embedding shape:", embedding.shape)  # Should be (768,)