# 🎙️ Classificador de Emoções em Áudio

Este projeto utiliza técnicas de **Machine Learning** para reconhecer emoções humanas a partir de áudios curtos com voz. A aplicação extrai **features acústicas** dos arquivos `.wav` e classifica a emoção predominante, como *raiva*, *felicidade*, *tristeza* ou *neutro*.


## 📁 Estrutura do Projeto

**audios**

Contém os arquivos .wav que serão usados para teste do modelo.

Exemplo de arquivos: audioteste1.wav, audioteste2.wav.

**codigo_fonte**

acoustic_features.csv: base de dados usada para treinar o modelo.

classificador_emocoes.py: script responsável por treinar e salvar o modelo.

**modelo**

Contém o modelo de Machine Learning já treinado.

Arquivo incluído:

modelo_emocoes.pkl: modelo salvo com a biblioteca Joblib.

**testes**

Contém os scripts de teste do modelo.

Arquivos incluídos:

teste_modelo.py: script para testar o modelo com a base CSV.

teste_real.py: script para testar o modelo com áudios reais.

**requirements.txt**

Lista de todas as dependências do projeto, utilizadas para instalação com pip.

**.gitignore**

Arquivo que define o que deve ser ignorado pelo Git).



## 🤖 Tecnologias Utilizadas

- Python 3.10+
- Librosa — extração de features de áudio
- Scikit-learn — criação e uso do modelo de classificação
- Pandas & NumPy — manipulação de dados
- Joblib — salvamento e carregamento do modelo


 ## ▶️ Como rodar o projeto localmente

- Clone o repositório
   
git clone https://github.com/carlosemanueln/classificador_emocoes

- Acesse a pasta do projeto
  
cd classificador_emocoes

- Crie um ambiente virtual
  
python -m venv venv

- Ative o ambiente virtual
1. Windows:
venv\Scripts\activate
2. Linux/macOS:
source venv/bin/activate

- Instale as dependências
  
pip install -r requirements.txt

- Execute o script principal
  
python codigo_fonte/classificador_emocoes.py


## 🎧 Como testar com seus próprios áudios

1. Coloque seus arquivos .wav na pasta audios/.
2. No script teste_real.py (em testes/), altere a variável com o nome do seu arquivo de áudio.
3. Execute o script para ver a previsão da emoção.


## 📊 Resultados Esperados

Após treinar o modelo com um conjunto de dados contendo características acústicas, ele é capaz de prever emoções com uma acurácia satisfatória. Os resultados variam conforme a qualidade do áudio e a clareza da fala.

⚠️ Observação importante:
O modelo não analisa o conteúdo do que é falado, mas sim características acústicas como entonação, ritmo e intensidade da voz. Ele é sensível à forma como a pessoa fala, não às palavras que ela usa.

Exemplo de saída: **Emoção prevista: felicidade**


## 📬 Contato
LinkedIn - www.linkedin.com/in/carlosemanuelp

Email: Cenbarreto.p@gmail.com
