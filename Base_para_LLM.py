# Dicionário para mapear palavras para índices
word_to_index = {
'Olá': 1,
'mundo': 2,
'como': 3,
'você': 4,
'está': 5
}
#A função de encode vai receber uma lista de palavras e converterá cada palavra
#em seu índice correspondente usando o dicionário.
def encode(text):
    return [word_to_index[word] for word in text.split() if word in word_to_index]

# Dicionário para mapear índices para palavras
index_to_word = {index: word for word, index in word_to_index.items()}

# A função de decode vai converter uma lista de índices de volta para uma
# string de palavras.
def decode(indices):
    return ' '.join(index_to_word[index] for index in indices if index in index_to_word)

# Texto de exemplo
text = "Olá mundo como você está"

# Codificação
encoded_text = encode(text)
print("Encoded:", encoded_text)

# Decodificação
decoded_text = decode(encoded_text)
print("Decoded:", decoded_text)