import joblib
import pandas as pd

# Carregando modelo
modelo = joblib.load('modelo_emocoes.pkl')
# Carregando os dados
df = pd.read_csv("Acoustic Features.csv")

# Entrada e Saída
X = df.drop("Class", axis=1)
y = df["Class"]

# Selecionando 5 entradas aleatórias
nova_entrada = X.sample(5, random_state=42) 

previsao = modelo.predict(nova_entrada)
print("Entradas selecionadas:")
print(nova_entrada)
print("\nEmoções previstas:")
print(previsao)