# O erro KeyError está ocorrendo porque o objeto train_encodings não é um dicionário com as chaves esperadas pelo método Trainer.

#O problema é que você está passando train_encodings como parâmetro para train_dataset e eval_dataset no construtor do Trainer. No entanto, train_encodings é um objeto BatchEncoding que não é um dicionário com as chaves esperadas pelo Trainer.

#Para corrigir o erro, você precisa criar um dicionário que contenha as entradas e labels para o treinamento e avaliação. Você pode fazer isso criando um dicionário com as chaves input_ids e attention_mask e passando-o como parâmetro para o Trainer.

# %%
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

# %%
# Carregar dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")
train_data = dataset['train']['text']

# %%
# Criar tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# %%
# Codificar dados
train_encodings = tokenizer(
    train_data,
    return_tensors='pt',
    max_length=1,
    truncation=True,
    padding='max_length'
)

# %%
# Criar configuração do modelo
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_ctx=512,
    n_embd=768,
    n_layer=12,
    n_head=12
)


# %%
# Criar modelo
model = GPT2LMHeadModel(config)

# %%
# Criar argumentos de treinamento
training_args = TrainingArguments(
    output_dir='./meuModeloGPT2',  
    num_train_epochs=5,  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,  
    evaluation_strategy='epoch',  
    learning_rate=5e-5,  
    save_total_limit=2,  
    save_strategy='epoch',  # Changed from 'teps' to 'epoch'
    load_best_model_at_end=True,  
    metric_for_best_model='loss',  
    greater_is_better=False,  
    eval_accumulation_steps=10,  
)

# %%
# Criar treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=train_encodings,
    compute_metrics=lambda pred: {'loss': pred.loss}  # Função para calcular métricas
)

# %%
# Treinar modelo
trainer.train()

# %%
# Criar pipeline de geração de texto
generator = pipeline('text-generation', model='./meuModeloGPT2', tokenizer='gpt2')

# %%
# Gerar texto
prompt = "Qual é o significado da vida, do universo e tudo mais?"
response = generator(prompt, max_length=50, num_return_sequences=1)
print(response[0]['generated_text'])  # Imprimir texto gerado



