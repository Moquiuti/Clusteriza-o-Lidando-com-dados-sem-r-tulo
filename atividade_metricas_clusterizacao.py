import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import joblib

url = 'https://raw.githubusercontent.com/alura-cursos/Clusterizacao-dados-sem-rotulo/main/Dados/dados_mkt.csv'

df = pd.read_csv(url)
df.head()

df.info()

df['sexo'].unique()

encoder = OneHotEncoder(categories=[['F', 'M', 'NE']], sparse_output=False)

encoded_sexo = encoder.fit_transform(df[['sexo']])
encoded_df = pd.DataFrame(
    encoded_sexo,
    columns=encoder.get_feature_names_out(['sexo'])
)

dados = pd.concat([df, encoded_df], axis=1).drop('sexo', axis=1)

dados.head()

joblib.dump(encoder, 'encoder.pkl')

mod_kmeans = KMeans(n_clusters=2, random_state=45, n_init='auto')
modelo = mod_kmeans.fit(dados)

mod_kmeans.inertia_

silhouette_score(dados, mod_kmeans.predict(dados))

def avaliacao(dados):
    inercia = []
    silhueta = []

    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=45, n_init='auto')
        kmeans.fit(dados)

        inercia.append(kmeans.inertia_)
        silhueta.append(
            f'k={k} - {silhouette_score(dados, kmeans.predict(dados))}'
        )

    return silhueta, inercia

silhueta, inercia = avaliacao(dados)
silhueta

def graf_silhueta(n_clusters, dados):
    kmeans = KMeans(n_clusters=n_clusters, random_state=45, n_init='auto')
    cluster_previsoes = kmeans.fit_predict(dados)

    silhueta_media = silhouette_score(dados, cluster_previsoes)
    print(f'Valor médio para {n_clusters} clusters: {silhueta_media:.3f}')

    silhueta_amostra = silhouette_samples(dados, cluster_previsoes)

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(9, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(dados) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhueta_amostra = silhueta_amostra[cluster_previsoes == i]
        ith_cluster_silhueta_amostra.sort()

        tamanho_cluster_i = ith_cluster_silhueta_amostra.shape[0]
        y_upper = y_lower + tamanho_cluster_i

        cor = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhueta_amostra,
            facecolor=cor,
            edgecolor=cor,
            alpha=0.7
        )

        ax1.text(-0.05, y_lower + 0.5 * tamanho_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.axvline(x=silhueta_media, color='red', linestyle='--')
    ax1.set_title(f'Gráfico da Silhueta para {n_clusters} clusters')
    ax1.set_xlabel('Valores do coeficiente de silhueta')
    ax1.set_ylabel('Rótulo do cluster')

    ax1.set_yticks([])
    ax1.set_xticks([i / 10.0 for i in range(-1, 11)])
    plt.show()

graf_silhueta(2, dados)

def plot_cotovelo(inercia):
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 21), inercia, 'bo-')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo para Determinação de k')
    plt.show()

plot_cotovelo(inercia)

df_resultado = df.copy()
df_resultado['cluster'] = modelo.labels_

joblib.dump(modelo, 'modelo_kmeans.pkl')
df_resultado.to_csv('dados_clusterizados.csv', index=False)

df_resultado.head()