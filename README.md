# Clusterização: Lidando com dados sem rótulo

Este repositório registra a atividade prática do curso **Clusterização: Lidando com dados sem rótulo**, com foco na preparação de dados categóricos e na criação de um modelo de agrupamento utilizando **K-Means**.

## Objetivo da atividade

A proposta da atividade foi construir um fluxo básico de Machine Learning não supervisionado para agrupar consumidores por conteúdos de interesse, considerando que o conjunto de dados não possui uma coluna de resposta esperada, ou seja, não possui rótulos.

## O que foi praticado

Durante a atividade foram realizadas as seguintes etapas:

1. Importação das bibliotecas necessárias;
2. Carregamento dos dados de marketing a partir de uma URL;
3. Análise inicial da estrutura do DataFrame;
4. Identificação da coluna categórica `sexo`;
5. Aplicação de codificação **One-Hot Encoding** para transformar dados textuais em valores numéricos;
6. Criação de um novo DataFrame preparado para o modelo;
7. Salvamento do encoder treinado com `joblib`;
8. Treinamento de um modelo de clusterização com **KMeans**;
9. Geração da coluna de cluster para identificar o grupo atribuído a cada consumidor;
10. Salvamento do modelo treinado e da base clusterizada.

## Conceitos envolvidos

### Dados sem rótulo

Neste tipo de problema, não existe uma resposta correta previamente conhecida. O modelo precisa identificar padrões nos dados com base nas características disponíveis.

### One-Hot Encoding

A técnica de One-Hot Encoding transforma categorias textuais em colunas binárias. No caso da coluna `sexo`, as categorias `F`, `M` e `NE` foram transformadas em colunas separadas, onde o valor `1` indica a presença daquela categoria e `0` indica ausência.

### Clusterização

Clusterização é uma técnica de aprendizado não supervisionado usada para agrupar dados semelhantes. O objetivo é encontrar grupos naturais dentro dos dados, mesmo sem uma classificação prévia.

### KMeans

O KMeans agrupa os dados em uma quantidade definida de clusters. Nesta atividade, foi utilizado `n_clusters=2`, com `random_state=45` para garantir reprodutibilidade.

## Arquivos gerados

- `atividade_clusterizacao.ipynb`: notebook principal da atividade;
- `requirements.txt`: dependências necessárias para executar o projeto;
- `README.md`: documentação da atividade;
- `encoder.pkl`: gerado após execução do notebook;
- `modelo_kmeans.pkl`: gerado após execução do notebook;
- `dados_clusterizados.csv`: gerado após execução do notebook.

## Como executar

Clone o repositório, instale as dependências e execute o notebook:

```bash
pip install -r requirements.txt
jupyter notebook atividade_clusterizacao.ipynb
```

Também é possível executar diretamente no Google Colab, copiando o conteúdo do notebook.

## Texto para entrega da atividade

Nesta atividade, trabalhei com um conjunto de dados de marketing sem rótulos, ou seja, sem uma resposta esperada previamente definida. O objetivo foi preparar os dados para um modelo de aprendizado não supervisionado capaz de agrupar consumidores com características semelhantes.

Inicialmente, realizei a importação das bibliotecas necessárias e carreguei o dataset utilizando o pandas. Em seguida, analisei a estrutura da base com `info()` e identifiquei que a coluna `sexo` possuía valores categóricos, como `F`, `M` e `NE`. Como modelos de machine learning trabalham melhor com dados numéricos, apliquei a técnica de One-Hot Encoding para transformar essa variável textual em colunas binárias.

Após essa etapa de preparação, concatenei os dados codificados ao DataFrame original, removendo a coluna textual `sexo`. Também salvei o encoder treinado com `joblib`, permitindo reutilizá-lo futuramente sem precisar ajustá-lo novamente.

Com os dados preparados, treinei um modelo de clusterização utilizando o algoritmo KMeans, configurado com dois agrupamentos e uma semente aleatória fixa para garantir reprodutibilidade. Por fim, adicionei ao DataFrame uma coluna indicando o cluster atribuído a cada registro, deixando a base pronta para análise dos grupos encontrados.

A atividade foi importante para reforçar o conceito de aprendizado não supervisionado, principalmente em cenários onde não existe uma resposta correta conhecida, mas sim a necessidade de identificar padrões escondidos nos dados.