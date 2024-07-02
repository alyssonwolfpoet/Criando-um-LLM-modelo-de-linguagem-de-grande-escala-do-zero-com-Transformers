import torch
from transformers import BertModel, BertTokenizer

# Carregar o modelo pré-treinado e o tokenizador
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Texto de exemplo
texto = "Exemplo de texto para codificar usando BERT."

# Tokenização
tokens = tokenizer.encode(texto, return_tensors='pt')

# Passar pelo modelo
outputs = model(tokens)

# Visualizar a saída
print(outputs.last_hidden_state.shape)  # Saída: torch.Size([1, 10, 768]) para BERT-base
