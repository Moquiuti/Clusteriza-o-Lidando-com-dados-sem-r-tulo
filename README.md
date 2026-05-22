# Clusterização: Lidando com dados sem rótulo

Este repositório contém **duas atividades separadas** do curso de clusterização:

1. **Atividade 1 - Pipeline base com KMeans**
2. **Atividade 2 - Avaliação de métricas do agrupamento**

## Estrutura das atividades

### Atividade 1 - Pipeline base com KMeans

Objetivo: preparar os dados e treinar o primeiro modelo de clusterização.

Arquivos:
- `atividade_clusterizacao.ipynb`
- `atividade_clusterizacao.py`

Conteudos praticados:
- carregamento dos dados de marketing;
- One-Hot Encoding da coluna `sexo`;
- treino inicial com `KMeans(n_clusters=2)`;
- geracao da coluna de cluster;
- salvamento de `encoder.pkl`, `modelo_kmeans.pkl` e `dados_clusterizados.csv`.

### Atividade 2 - Avaliando métricas do modelo

Objetivo: avaliar a qualidade dos agrupamentos com métricas e visualizações.

Arquivos:
- `atividade_metricas_clusterizacao.ipynb`
- `atividade_metricas_clusterizacao.py`

Conteudos praticados:
- calculo de inércia;
- calculo de Silhouette Score;
- comparacao de diferentes valores de `k` (2 a 20);
- grafico de silhueta;
- metodo do cotovelo.

## Dependências

As dependências para as duas atividades estão em `requirements.txt`.

## Como executar

### Opção 1: scripts Python

```bash
pip install -r requirements.txt
python atividade_clusterizacao.py
python atividade_metricas_clusterizacao.py
```

### Opção 2: notebooks

```bash
pip install -r requirements.txt
jupyter notebook atividade_clusterizacao.ipynb
jupyter notebook atividade_metricas_clusterizacao.ipynb
```

## Arquivos gerados na execução

- `encoder.pkl`
- `modelo_kmeans.pkl`
- `dados_clusterizados.csv`

Esses arquivos podem ser sobrescritos ao executar as atividades.
