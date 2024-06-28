"""
Sim, há mais melhorias possíveis no código:

   1 Utilizar Trainer do Transformers: Em vez de implementar um loop de treinamento manual, você pode utilizar o Trainer do Transformers, que é mais robusto e fácil de usar.
   2 Utilizar DataCollator do Transformers: Em vez de criar um DataLoader manualmente, você pode utilizar o DataCollator do Transformers, que é mais fácil de usar e mais eficiente.
   3 Utilizar Accelerator do Transformers: Em vez de treinar o modelo em um dispositivo específico, você pode utilizar o Accelerator do Transformers, que é mais fácil de usar e mais eficiente.
   4 Adicionar mais hiperparâmetros: Você pode adicionar mais hiperparâmetros para ajustar o modelo, como o tamanho do batch, o número de epochs, o learning rate, etc.
   5 Utilizar EarlyStopping: Você pode utilizar EarlyStopping para parar o treinamento quando o modelo atinge um certo nível de performance.
   6 Utilizar ModelCheckpoint: Você pode utilizar ModelCheckpoint para salvar o modelo em intervalos regulares durante o treinamento.
   7 Adicionar mais métricas de avaliação: Você pode adicionar mais métricas de avaliação, como a perda de validação, a precisão, a revocação, etc.
   8 Utilizar TensorBoard: Você pode utilizar TensorBoard para visualizar o treinamento do modelo e avaliar seu desempenho.
   9 Adicionar mais comentários e documentação: Você pode adicionar mais comentários e documentação para explicar o código e torná-lo mais fácil de entender.
   10 Utilizar Type Hints: Você pode utilizar Type Hints para especificar os tipos de variáveis e parâmetros, tornando o código mais legível e fácil de entender.

"""
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from transformers import pipeline

# Load dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")
train_data = dataset['train']['text']

# Create tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Encode data
train_encodings = tokenizer(train_data, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

# Create model config and model
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_ctx=512,
    n_embd=768,
    n_layer=12,
    n_head=12
)
model = GPT2LMHeadModel(config)

# Create training arguments
training_args = TrainingArguments(
    output_dir='./meuModeloGPT2',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    save_total_limit=2,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    greater_is_better=False,
    save_strategy='steps',
    eval_accumulation_steps=10,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=train_encodings,
    compute_metrics=lambda pred: {'loss': pred.loss},
)

# Train model
trainer.train()

# Create generator pipeline
generator = pipeline('text-generation', model='./meuModeloGPT2', tokenizer='gpt2')

# Generate text
prompt = "Qual é o significado da vida, do universo e tudo mais?"
response = generator(prompt, max_length=50, num_return_sequences=1)
print(response[0]['generated_text'])