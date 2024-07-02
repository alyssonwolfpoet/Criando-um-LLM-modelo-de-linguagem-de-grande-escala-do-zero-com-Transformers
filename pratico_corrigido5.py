# Importar bibliotecas necessárias
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar dados do arquivo CSV
dataset = pd.read_csv('your_data.csv')

# Separar dados em conjuntos de treinamento e avaliação
train_text, eval_text, train_labels, eval_labels = train_test_split(dataset['text'], dataset['label'], test_size=0.2, random_state=42)

# Definir modelo e tokenizer
model_name = 'neuralmind/bert-base-portuguese-cased'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Função para codificar texto em embeddings
def encode(texts):
    return tokenizer(texts, return_tensors='pt', truncation=True, padding=True)

# Codificar dados de treinamento
train_encodings = encode(train_text)

# Codificar dados de avaliação
eval_encodings = encode(eval_text)

# Converter labels em tensores
train_labels_tensor = torch.tensor(train_labels)
eval_labels_tensor = torch.tensor(eval_labels)

# Definir dispositivo (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Definir critério de perda e otimizador
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Treinar modelo
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    outputs = model(**train_encodings.to(device))
    loss = criterion(outputs.logits, train_labels_tensor.to(device))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Avaliar modelo
model.eval()
eval_outputs = model(**eval_encodings.to(device))
eval_logits = eval_outputs.logits.detach().cpu().numpy()
eval_preds = torch.argmax(torch.tensor(eval_logits), dim=1).numpy()
print(f'Acurácia: {accuracy_score(eval_labels, eval_preds)}')