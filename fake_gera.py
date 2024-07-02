import pandas as pd

from faker import Faker


fake = Faker('pt_BR')


data = []

for i in range(100):

    text = fake.text(max_nb_chars=200)

    label = 'positivo' if fake.random_int(0, 1) == 0 else 'negativo'

    data.append({'text': text, 'label': label})


df = pd.DataFrame(data)

df.to_csv('your_data.csv', index=False)