import os
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zmq import NULL
from PIL import Image
from wordcloud import WordCloud
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
#nltk.download("all")

textos_verdadeiros_treino = []
textos_falsos_treino = []


# carrega as pastas dos arquivos e cria lista vazia "data"
main_folder = 'full_texts'
ntexts_folder = 'size_normalized_texts'
data = []

def dataframe_processing():

    def process_folder(folder, tag):
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    number = os.path.splitext(filename)[0]
                    data.append([f"{number}-{tag}", tag, text])

    # Processa as pastas "True" e "Fake":
    true_folder = os.path.join(main_folder, "true")
    fake_folder = os.path.join(main_folder, "fake")
    process_folder(true_folder, "True")
    process_folder(fake_folder, "Fake")

    # Cria um dataframe com os dados coletados
    df_full_texts = pd.DataFrame(data, columns=["Id", "Tag", "full_text"])
    # Configura o Id como index
    df_full_texts.set_index("Id", inplace=True)

    # Cria listas para nome de pastas e colunas dos metadados e lista metadata vazia
    metadata_folders = ['true-meta-information', 'fake-meta-information']
    column_names = [
        "Id", "author", "link", "category", "date of publication",
        "number of tokens", "number of words without punctuation",
        "number of types", "number of links inside the news",
        "number of words in upper case", "number of verbs",
        "number of subjunctive and imperative verbs", "number of nouns",
        "number of adjectives", "number of adverbs",
        "number of modal verbs (mainly auxiliary verbs)",
        "number of singular first and second personal pronouns",
        "number of plural first personal pronouns", "number of pronouns",
        "pausality", "number of characters", "average sentence length",
        "average word length", "percentage of news with spelling errors",
        "emotiveness", "diversity"
    ]
    metadata = []

    # Função que abre o arquivo, extrai o texto e cria o Dataframe extraindo o Id da pasta e nome do arquivo
    def process_folder_meta(folder, tag):
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                    lines = file.read().split('\n')
                    number = filename.split('-')[0]
                    row_data = [f"{number}-{tag}"]
                    row_data.extend(lines)  # Add the content from the file
                    metadata.append(row_data)

    # Processa as pastas "true-meta-information" e "fake-meta-information":
    true_meta_folder = os.path.join(main_folder, 'true-meta-information')
    fake_meta_folder = os.path.join(main_folder, 'fake-meta-information')
    process_folder_meta(true_meta_folder, "True")
    process_folder_meta(fake_meta_folder, "Fake")

    # Cria um dataframe com os dados coletados e exclui "Id" da lista column_names
    df_metadados = pd.DataFrame(metadata, columns=["Id"] + column_names[1:])
    # Configura o Id como index
    df_metadados.set_index("Id", inplace=True)

    # Cria lista vazia data_ntext
    data_ntext = []

    # Função que abre o arquivo, extrai o texto e cria o Dataframe extraindo o Id da pasta e nome do arquivo
    def process_folder_n(folder, tag):
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    number = os.path.splitext(filename)[0]
                    data_ntext.append([f"{number}-{tag}", text])

    # Processa as pastas "True" e "Fake":
    true_folder = os.path.join(ntexts_folder, "true")
    fake_folder = os.path.join(ntexts_folder, "fake")
    process_folder_n(true_folder, "True")
    process_folder_n(fake_folder, "Fake")

    # Cria um dataframe com os dados coletados
    df_ntext = pd.DataFrame(data_ntext, columns=["Id", "normalized_text"])
    # Configura o Id como index
    df_ntext.set_index("Id", inplace=True)

    # Cria o Dataframe df_geral_corpus concatenando os outros dataframes
    df_geral_corpus = pd.concat([df_full_texts, df_ntext, df_metadados], axis=1)
    # Preenche campos "NaN" com "0"
    df_geral_corpus = df_geral_corpus.fillna(value=NULL)

    print(df_geral_corpus)

    # Salva o dataframe em um arquivo CSV no Google Drive
    df_geral_corpus.to_csv('df_geral_corpus.csv', index=False)



def clean_train():
    global textos_verdadeiros_treino, textos_falsos_treino
    df_geral_csv = pd.read_csv('df_geral_corpus.csv')
    lemmatizer = WordNetLemmatizer()  # Inicializa o lematizador

    # Função para pré-processamento com lematização
    def limpeza_texto(texto_pre):
        tokens = word_tokenize(texto_pre, language='portuguese')  # tokenização
        tokens = [unidecode(token).lower() for token in tokens if
                  token.isalpha()]  # remove acento e p/ minúsculas
        stop_words = set(stopwords.words('portuguese'))  # cria lista de stopwords
        tokens = [token for token in tokens if
                  token not in stop_words and token not in string.punctuation]  # remove stopwords
        tokens = [lemmatizer.lemmatize(token, wordnet.VERB) for token in tokens]  # Lematização
        texto_processado = " ".join(tokens)
        return texto_processado

    # Aplica a função de pré-processamento ao DataFrame
    df_geral_csv['processado'] = df_geral_csv['normalized_text'].apply(limpeza_texto)

    # Mapeia 'Tag' para True (verdadeiro) e Fake (falso)
    df_geral_csv['Tag'] = df_geral_csv['Tag'].map({'True': True, 'Fake': False})
    # Separa os textos verdadeiros
    textos_verdadeiros = df_geral_csv[df_geral_csv['Tag'] == True]['processado']
    # Separa os textos falsos
    textos_falsos = df_geral_csv[df_geral_csv['Tag'] == False]['processado']
    # Divisão os textos verdadeiros em 75% para treinamento e 25% para teste
    textos_verdadeiros_treino, textos_verdadeiros_teste = train_test_split(textos_verdadeiros,
                                                                           test_size=0.25,
                                                                           random_state=42)
    # Divisão dos textos falsos em 75% para treinamento e 25% para teste
    textos_falsos_treino, textos_falsos_teste = train_test_split(textos_falsos, test_size=0.25,
                                                                 random_state=42)

    #treina o modelo
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Cria um vetorizador TF-IDF
    # Vetoriza os textos verdadeiros e falsos de treinamento
    X_treino = tfidf_vectorizer.fit_transform(
        list(textos_verdadeiros_treino) + list(textos_falsos_treino))
    # Cria os rótulos para os textos (True para textos verdadeiros e False para textos falsos)
    y_treino = [True] * len(textos_verdadeiros_treino) + [False] * len(textos_falsos_treino)
    clf = LogisticRegression(solver="lbfgs")  # inicializa classif. regr. log. lbfgs
    clf.fit(X_treino, y_treino)  # treina o modelo
    # Vetoriza os textos de teste
    X_teste = tfidf_vectorizer.transform(list(textos_verdadeiros_teste) + list(textos_falsos_teste))
    # Cria os rótulos para os textos de teste
    y_teste = [True] * len(textos_verdadeiros_teste) + [False] * len(textos_falsos_teste)
    previsoes = clf.predict(X_teste)  # Faz previsões usando o mod. treinado
    acuracia = accuracy_score(y_teste, previsoes)  # calcula a acurácia
    print("Acurácia: {:.2f}%".format(acuracia * 100))  # mostra o valor acurácia

# Função Gera Num de palabras em formato de joinha \ que cria com as características de máscara, cor,
# contorno
def criar_nuvem_de_palavras(texto_completo, mask_image, colormap, contour_color, contour_width):
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        mask=mask_image,
        colormap=colormap,
        contour_color=contour_color,
        contour_width=contour_width
    ).generate(texto_completo)

    # Plota a nuvem de palavras
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show(block=False)  # Non-blocking show
    plt.pause(3)  # Display the plot for 3 seconds
    plt.close()  # Close the plot window

# função vetoriza, que conta palavras, digramas e trigramas
def vetoriza_tfidf(textos):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))# Crie um vetorizador TF-IDF
    # Vetoriza os textos
    X = tfidf_vectorizer.fit_transform(textos)
    # extrai o vocabulário de palavras, bigramas e trigramas
    vocab = tfidf_vectorizer.get_feature_names_out()
    # Número total de palavras, bigramas e trigramas
    total_palavras = len(vocab)
    # Iteração para contar bigramas e trigramas
    total_bigramas = sum(1 for palavra in vocab if palavra.count(' ') == 1)
    total_trigramas = sum(1 for palavra in vocab if palavra.count(' ') == 2)
    # Imprime os resultados
    print("Total de palavras:", total_palavras)
    print("Total de bigramas:", total_bigramas)
    print("Total de trigramas:", total_trigramas)

    return X

def generate_true():
    global textos_verdadeiros_treino
    # Concatenar todos os textos verdadeiros em um único texto / carrega a imagem like / chama a funçao de criar nuvem
    texto_verdadeiro_completo = " ".join(textos_verdadeiros_treino)
    mask_like = np.array(Image.open("like_shape.png"))
    criar_nuvem_de_palavras(texto_verdadeiro_completo, mask_like, 'Greens', 'green', 1)

    # Computa os textos verdadeiros de treinamento e imprime resultados de contagem:
    X_treino_verdadeiro = vetoriza_tfidf(textos_verdadeiros_treino)

def generate_false():
    global textos_falsos_treino
    texto_falso_completo = " ".join(textos_falsos_treino)
    mask_dislike = np.array(Image.open("dislike_shape.png"))
    criar_nuvem_de_palavras(texto_falso_completo, mask_dislike, 'Reds', 'red', 1)

    # Computa os textos falsos de treinamento e imprime resultados de contagem:
    X_treino_falso = vetoriza_tfidf(textos_falsos_treino)


#processa o df do corpus
dataframe_processing()

#limpa, normaliza e treina o modelo, mostrando a eficácia
clean_train()

#gera resultados de textos verdadeiros = formato de like e números de n-gramas
generate_true()

#gera resultados de textos falsos = formato de dislike e números de n-gramas
generate_false()




