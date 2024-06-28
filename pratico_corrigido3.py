"""Melhorias adicionadas:

    Adicionado comentários em pt-br para explicar o código
    Utilizado TrainingArguments para definir argumentos de treinamento
    Utilizado Trainer para treinar o modelo
    Adicionado compute_metrics para calcular métricas durante o treinamento
    Utilizado pipeline para criar uma pipeline de geração de texto
    Adicionado num_return_sequences para especificar o número de sequências a serem geradas

Essas são apenas algumas sugestões de melhorias, e há muitas outras coisas que você pode fazer para melhorar o código."""

# Importar bibliotecas necessárias
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    pipeline
)

# Carregar dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")
train_data = dataset['train']['text']

# Criar tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Codificar dados
train_encodings = tokenizer(
    train_data,
    return_tensors='pt',
    max_length=512,
    truncation=True,
    padding='max_length'
)

# Criar configuração do modelo
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_ctx=512,
    n_embd=768,
    n_layer=12,
    n_head=12
)

# Criar modelo
model = GPT2LMHeadModel(config)

# Criar argumentos de treinamento
training_args = TrainingArguments(
    output_dir='./meuModeloGPT2',  # Diretório de saída do modelo
    num_train_epochs=5,  # Número de épocas de treinamento
    per_device_train_batch_size=8,  # Tamanho do batch por dispositivo
    per_device_eval_batch_size=8,  # Tamanho do batch por dispositivo para avaliação
    evaluation_strategy='epoch',  # Estratégia de avaliação
    learning_rate=5e-5,  # Taxa de aprendizado
    save_total_limit=2,  # Número de modelos a serem salvos
    save_steps=500,  # Passos para salvar o modelo
    load_best_model_at_end=True,  # Carregar o melhor modelo ao final do treinamento
    metric_for_best_model='loss',  # Métrica para selecionar o melhor modelo
    greater_is_better=False,  # Se a métrica é melhor quando é maior
    save_strategy='steps',  # Estratégia de salvamento
    eval_accumulation_steps=10,  # Passos para acumular avaliações
)

# Criar treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=train_encodings,
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