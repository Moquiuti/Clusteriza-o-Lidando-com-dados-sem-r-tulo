# Auto-generated from atividade_clusterizacao.ipynb

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import joblib

# %%
url = 'https://raw.githubusercontent.com/alura-cursos/Clusterizacao-dados-sem-rotulo/main/Dados/dados_mkt.csv'

df = pd.read_csv(url)
df.head()

# %%
df.info()

# %%
df['sexo'].unique()

# %%
encoder = OneHotEncoder(categories=[['F', 'M', 'NE']], sparse_output=False)

encoded_sexo = encoder.fit_transform(df[['sexo']])
encoded_df = pd.DataFrame(
    encoded_sexo,
    columns=encoder.get_feature_names_out(['sexo'])
)

encoded_df.head()

# %%
dados = pd.concat([df, encoded_df], axis=1).drop('sexo', axis=1)

dados.head()

# %%
joblib.dump(encoder, 'encoder.pkl')

# %%
mod_kmeans = KMeans(n_clusters=2, random_state=45, n_init='auto')
modelo = mod_kmeans.fit(dados)

modelo

# %%
df_resultado = df.copy()
df_resultado['cluster'] = modelo.labels_

df_resultado.head()

# %%
df_resultado['cluster'].value_counts()

# %%
score = silhouette_score(dados, modelo.labels_)
print(f'Silhouette Score: {score:.4f}')

# %%
joblib.dump(modelo, 'modelo_kmeans.pkl')
df_resultado.to_csv('dados_clusterizados.csv', index=False)
