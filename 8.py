import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import pipeline

# Load dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")
train_data = dataset['train']['text']

# Create tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Encode data
train_encodings = tokenizer(train_data, return_tensors='pt', max_length=4, truncation=True, padding="max_length")

# Create model config and model
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel(config)

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create optimizer and data loader
optimizer = AdamW(model.parameters(), lr=5e-5)
train_loader = DataLoader(train_encodings, batch_size=8, shuffle=True)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss calculation

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss {total_loss / len(train_loader)}")

# Save model
model.save_pretrained('./meuModeloGPT2')

# Create generator pipeline
generator = pipeline('text-generation', model='./meuModeloGPT2', tokenizer='gpt2')

# Generate text
prompt = "Qual é o significado da vida, do universo e tudo mais?"
response = generator(prompt, max_length=50, num_return_sequences=1)
print(response[0]['generated_text'])