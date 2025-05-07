# Biblioteca para manipulação de DataFrame
import pandas as pd

# Carregando o dataset
data = pd.read_csv('Acoustic Features.csv')

# Conhecendo as colunas e dados do dataset
data.columns

# Verificando se há valores ausentes
data.isnull().sum()


#---------------------------------------------------------------------
## A coluna "Class" é a emoção. 
## As outras colunas são variáveis de entrada.
## Não há valores ausentes.
#---------------------------------------------------------------------


# =======================
##  TREINANDO O MODELO
# =======================


from sklearn.model_selection import train_test_split  # Para dividir os dados em treino e teste
from sklearn.ensemble import RandomForestClassifier   # Algoritmo de classificação
from sklearn.metrics import classification_report, accuracy_score  # Métricas de avaliação

# Separa os dados de entrada (X)
X = data.drop('Class', axis=1)

# Coluna 'Class'
y = data['Class']


# Dividindo os dados em conjunto de treino (80%) e teste (20%)
# random_state garante que a divisão seja sempre igual (reprodutível)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Criando o modelo de classificação Random Forest
model = RandomForestClassifier(random_state=42)

# Treinando o modelo com os dados de treino
model.fit(X_train, y_train)

# Previsões com os dados de teste
y_pred = model.predict(X_test)


# Avaliando quantas previsões foram corretas
# Calculo de acurácia padrão: Acertos / Total de previsões
print("Acurácia do modelo:")
print(accuracy_score(y_test, y_pred))


# Relatório completo para o caso de desbalanceamento na divisão (treino x teste)
# Precisão, recall e F1-score para cada emoção
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))



# =======================
## TREINANDO MODELO DE CLASSIFICAÇÃO
## RANDOM FOREST E ADIÇÃO DE COLUNA COM AS PREVISÕES DO MODELO
## TABELA COM DADOS REAIS E AS PREVISÕES DO MODELO
# =======================


df = pd.read_csv("Acoustic Features.csv")
# Separando as colunas de entrada e a saída
X = df.drop("Class", axis=1)  # X recebe todas as colunas, exceto "Class"
y = df["Class"]               # y recebe somente a coluna "Class"

# Dividindo os dados em conjunto de treino (80%) e teste (20%)
# O random_state garante que a divisão será sempre igual em todas as execuções
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo Random Forest
# n_estimators define quantas árvores vão compor a floresta
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinando o modelo com os dados de treino
model.fit(X_train, y_train)


# Fazendo previsões com os dados de teste
y_pred = model.predict(X_test)


# Avaliando a acurácia das previsões
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo:", accuracy)


# Mostrando um relatório mais completo de métricas
report = classification_report(y_test, y_pred)
print("Relatório de classificação:\n", report)


# =======================
## RESULTADOS COM A MATRIZ DE CONFUSÃO DO MODELO
# =======================


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Criando a matriz: rótulos reais X previstos pelo classificador
cm = confusion_matrix(y_test, y_pred)

# Criando um gráfico da matriz
plt.figure(figsize=(8, 6))  # Tamanho do gráfico
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)

# Rótulos
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()


import joblib
joblib.dump(model,'modelo_emocoes.pkl')