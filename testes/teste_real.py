import librosa  # Carregar o áudio e extrair características acústicas
import numpy as np 
import joblib 
import os  # Para manipular caminhos de arquivos

# ==========================================
## FUNÇÃO DE EXTRACAO
# ==========================================
def extrair_features(caminho_audio):
    # 'y' é o sinal de áudio e 'sr' é a taxa de amostragem
    y, sr = librosa.load(caminho_audio, sr=None)
    
    ## Características acústicas do áudio:
    
    # Taxa de zero-crossings
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T)
    
    # Energia média
    energia = np.mean(librosa.feature.rms(y=y).T)
    
    # Centróide espectral
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T)
    
    # Rolloff espectral
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T)
    
    # Coeficientes MFCC (Frequência)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Média do MFCC
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Aqui é importante que você extrair as 50 características usadas no treinamento
    # Vamos considerar que as 50 características são uma combinação de várias outras características
    # como energia, MFCC, etc. Certifique-se de ajustar isso para corresponder às características exatas
    # usadas no seu treinamento.
    
    # Exemplo: Se o modelo usou outras características além dessas, você precisa extraí-las aqui.

    # Vamos criar um vetor de 50 características, preenchendo com valores fictícios (se não houver outras extrações).
    # Atualize com as reais características usadas no seu treinamento.
    
    # Junta todas as características em um único vetor
    features = np.hstack([zcr, energia, centroid, rolloff, mfccs_mean])  # Aqui só temos 17 características
    
    # Preenche o vetor para ter 50 características (usando valores fictícios)
    features = np.pad(features, (0, 50 - len(features)), mode='constant')
    
    # Retorna o vetor de características
    return features.reshape(1, -1)  # Reshape para o formato esperado pelo modelo

# ============================= #
# CAMINHO DO ÁUDIO E CARREGAMENTO DO MODELO #
# ============================= #
# Caminho dos arquivos de áudio — você pode substituir conforme necessário
arquivos_audio = ["audioteste1.wav", "audioteste2.wav"]  # Lista de arquivos de áudio para teste

# Verifica se os arquivos existem na pasta
for arquivo_audio in arquivos_audio:
    if not os.path.exists(arquivo_audio):
        print(f"Arquivo de áudio '{arquivo_audio}' não encontrado.")
    else:
        # Carrega o modelo treinado
        modelo = joblib.load("modelo_emocoes.pkl")
        
        # Extrai as características do áudio
        entrada = extrair_features(arquivo_audio)
        
        # Faz a previsão da emoção
        emocao_prevista = modelo.predict(entrada)
        
        # Mostra o resultado da previsão
        print(f"Emoção prevista para '{arquivo_audio}': {emocao_prevista[0]}")
