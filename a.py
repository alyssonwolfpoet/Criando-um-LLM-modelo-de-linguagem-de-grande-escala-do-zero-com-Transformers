from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Carregar o dataset wikitext-103
dataset = load_dataset("wikitext", "wikitext-103-v1")
train_data = dataset['train']['text']

# Inicializar o tokenizador GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Adicionar token especial de padding se não estiver configurado
if tokenizer.pad_token is None:
     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenizar os dados de treinamento com padding
train_encodings = tokenizer(train_data, return_tensors='pt', max_length=512, truncation=True, padding=True)

# Configurar o modelo GPT-2
config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        n_ctx=512,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
model = GPT2LMHeadModel(config)

# Inicializar o otimizador
optimizer = AdamW(model.parameters(), lr=5e-5)
    
# Carregar os dados de treinamento usando DataLoader
train_loader = DataLoader(train_encodings, batch_size=8, shuffle=True)

    # Colocar o modelo em modo de treinamento
model.train()

# Treinamento do modelo
for epoch in range(5):
    for batch in train_loader:
            inputs, labels = batch['input_ids'], batch['input_ids']
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    print(f"Epoch {epoch}: Loss {loss.item()}")

# Salvar o modelo treinado
model.save_pretrained('./meuModeloGPT2')

# Carregar o modelo treinado para geração de texto
generator = GPT2LMHeadModel.from_pretrained('./meuModeloGPT2')

# Exemplo de geração de texto
prompt = "Qual é o significado da vida, do universo e tudo mais?"
generated_text = generator.generate(prompt, max_length=50, num_return_sequences=1)
print(generated_text)