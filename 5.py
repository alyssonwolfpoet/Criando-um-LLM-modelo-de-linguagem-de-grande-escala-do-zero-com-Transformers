# Importar bibliotecas necessárias para o projeto
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,  # Tokenizador para o modelo GPT-2
    GPT2Config,  # Configuração do modelo GPT-2
    GPT2LMHeadModel,  # Modelo GPT-2 com cabeçalho de linguagem
    Trainer,  # Treinador do modelo
    TrainingArguments,  # Argumentos de treinamento do modelo
    pipeline  # Pipeline para geração de texto
)

# Carregar dataset de treinamento
dataset = load_dataset("wikitext", "wikitext-103-v1")
train_data = dataset['train']['text']  # Selecionar dados de treinamento

# Criar tokenizer para o modelo GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Definir token de padding

# Codificar dados de treinamento
train_encodings = tokenizer(
    train_data,
    return_tensors='pt',  # Retorna tensors PyTorch
    max_length=512,  # Tamanho máximo da sequência
    truncation=True,  # Truncar sequências longas
    padding='max_length'  # Preencher sequências curtas com padding
)

# Criar configuração do modelo GPT-2
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,  # Tamanho do vocabulário
    n_positions=512,  # Número de posições na sequência
    n_ctx=512,  # Número de contextos na sequência
    n_embd=768,  # Número de dimensões do embedding
    n_layer=12,  # Número de camadas do modelo
    n_head=12  # Número de cabeçalhos de atenção
)

# Criar modelo GPT-2 com cabeçalho de linguagem
model = GPT2LMHeadModel(config)

# Criar argumentos de treinamento do modelo
training_args = TrainingArguments(
    output_dir='./meuModeloGPT2',  # Diretório de saída do modelo
    num_train_epochs=5,  # Número de épocas de treinamento
    per_device_train_batch_size=8,  # Tamanho do batch de treinamento por dispositivo
    per_device_eval_batch_size=8,  # Tamanho do batch de avaliação por dispositivo
    evaluation_strategy='epoch',  # Estratégia de avaliação por época
    learning_rate=5e-5,  # Taxa de aprendizado
    save_total_limit=2,  # Número de modelos salvos
    save_strategy='epoch',  # Estratégia de salvamento por época
    load_best_model_at_end=True,  # Carregar melhor modelo ao final do treinamento
    metric_for_best_model='loss',  # Métrica para selecionar melhor modelo
    greater_is_better=False,  # Se a métrica é melhor quando é maior
    eval_accumulation_steps=10  # Número de passos de avaliação por época
)

# Criar dataset para treinamento
train_dataset = {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']}

# Criar treinador do modelo
trainer = Trainer(
    model=model,  # Modelo a ser treinado
    args=training_args,  # Argumentos de treinamento
    train_dataset=train_dataset,  # Dataset de treinamento
    eval_dataset=train_dataset,  # Dataset de avaliação
    compute_metrics=lambda pred: {'loss': pred.loss}  # Função para calcular métricas
)

# Treinar modelo
trainer.train()

# Criar pipeline de geração de texto
generator = pipeline('text-generation', model='./meuModeloGPT2', tokenizer='gpt2')

# Gerar texto
prompt = "Qual é o significado da vida, do universo e tudo mais?"
response = generator(prompt, max_length=50, num_return_sequences=1)
print(response[0]['generated_text'])  # Imprimir texto gerado