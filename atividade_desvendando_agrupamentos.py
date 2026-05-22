import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans

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

scaler = MinMaxScaler()
dados_escalados = scaler.fit_transform(dados)

dados_escalados = pd.DataFrame(
    dados_escalados,
    columns=dados.columns
)

modelo_kmeans = KMeans(n_clusters=3, random_state=45, n_init='auto')
modelo_kmeans.fit(dados_escalados)

joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(modelo_kmeans, 'kmeans.pkl')

dados_analise = pd.DataFrame()
dados_analise[dados_escalados.columns] = scaler.inverse_transform(dados_escalados)
dados_analise.head()

dados_analise['cluster'] = modelo_kmeans.labels_

cluster_media = dados_analise.groupby('cluster').mean()
cluster_media.T

cluster_media = cluster_media.transpose()
cluster_media.columns = [0, 1, 2]

cluster_media[0].sort_values(ascending=False)
cluster_media[1].sort_values(ascending=False)
cluster_media[2].sort_values(ascending=False)

dados_analise.to_csv('dados_analise_clusterizados.csv', index=False)
cluster_media.to_csv('medias_por_cluster.csv')

resumo_clusters = {
    0: 'Público jovem com forte interesse em moda, música e aparência.',
    1: 'Público associado a esportes, especialmente futebol americano, basquete e atividades culturais como banda e rock.',
    2: 'Público mais equilibrado, com interesses em música, dança e moda.'
}

for cluster, descricao in resumo_clusters.items():
    print(f'Grupo {cluster}: {descricao}')
