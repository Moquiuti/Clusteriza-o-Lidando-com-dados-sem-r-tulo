# Clusterização: Lidando com dados sem rótulo

Este repositório reúne a evolução prática do curso de clusterização, organizado em **três atividades complementares**:

1. **Atividade 1 - Pipeline base com KMeans**
2. **Atividade 2 - Avaliação de métricas de clusterização**
3. **Atividade 3 - Otimização do resultado dos agrupamentos**

## Visão geral das atividades

### Atividade 1 - Pipeline base com KMeans

**Objetivo**: preparar a base e treinar o primeiro modelo de agrupamento.

**Arquivos**:
- `atividade_clusterizacao.ipynb`
- `atividade_clusterizacao.py`

**Práticas aplicadas**:
- carregamento da base de marketing;
- codificação da variável categórica `sexo` com One-Hot Encoding;
- treinamento inicial com `KMeans(n_clusters=2)`;
- geração da coluna de cluster;
- salvamento de `encoder.pkl`, `modelo_kmeans.pkl` e `dados_clusterizados.csv`.

### Atividade 2 - Avaliação de métricas de clusterização

**Objetivo**: avaliar a qualidade dos agrupamentos com métricas e análises visuais.

**Arquivos**:
- `atividade_metricas_clusterizacao.ipynb`
- `atividade_metricas_clusterizacao.py`

**Práticas aplicadas**:
- cálculo de inércia;
- cálculo de Silhouette Score;
- comparação de diferentes valores de `k` (2 a 20);
- gráfico de silhueta;
- método do cotovelo.

### Atividade 3 - Otimizando o resultado dos agrupamentos

**Objetivo**: melhorar o desempenho do KMeans aplicando escalonamento com `MinMaxScaler` e reavaliando as métricas.

**Arquivos**:
- `atividade_otimizacao_clusterizacao.ipynb`
- `atividade_otimizacao_clusterizacao.py`

**Práticas aplicadas**:
- carregamento e preparação da base;
- One-Hot Encoding da coluna `sexo`;
- análise estatística com `describe()`;
- escalonamento com `MinMaxScaler`;
- reconstrução dos dados escalados em `DataFrame`;
- salvamento do scaler treinado com `joblib`;
- avaliação de inércia e silhueta para diferentes valores de `k`;
- gráfico de silhueta para 3 clusters;
- gráfico do método do cotovelo;
- treinamento final com `KMeans(n_clusters=3)`;
- salvamento do modelo e da base final clusterizada.

## Conceitos trabalhados

### One-Hot Encoding

Transforma variáveis categóricas em colunas binárias para permitir o uso em modelos numéricos.

### Inércia

Mede a soma das distâncias dos pontos ao centróide do seu cluster. É útil para comparação entre modelos, mas tende a diminuir com o aumento de `k`.

### Silhouette Score

Avalia a separação entre clusters e a coesão interna dos grupos:
- valores próximos de `1`: melhor separação;
- valores próximos de `0`: sobreposição entre grupos;
- valores próximos de `-1`: possível alocação inadequada de pontos.

### Escalonamento com MinMaxScaler

Padroniza os atributos para o intervalo `[0, 1]`, reduzindo o impacto de variáveis com escalas maiores em algoritmos baseados em distância, como KMeans.

## Dependências

As dependências necessárias para todas as atividades estão em `requirements.txt`.

## Como executar

### Scripts Python

```bash
pip install -r requirements.txt
python atividade_clusterizacao.py
python atividade_metricas_clusterizacao.py
python atividade_otimizacao_clusterizacao.py
```

### Notebooks

```bash
pip install -r requirements.txt
jupyter notebook atividade_clusterizacao.ipynb
jupyter notebook atividade_metricas_clusterizacao.ipynb
jupyter notebook atividade_otimizacao_clusterizacao.ipynb
```

## Artefatos gerados

Dependendo da atividade executada, os seguintes arquivos podem ser gerados ou atualizados:

- `encoder.pkl`
- `scaler.pkl`
- `modelo_kmeans.pkl`
- `kmeans.pkl`
- `dados_clusterizados.csv`
- `dados_clusterizados_escalados.csv`

## Texto para entrega - Atividade 3

Nesta atividade, dei continuidade ao fluxo de clusterização buscando otimizar o resultado dos agrupamentos gerados pelo modelo KMeans. Depois de preparar os dados e aplicar a codificação One-Hot Encoding na coluna categórica `sexo`, utilizei o método `describe()` para observar as estatísticas resumidas da base e entender melhor a distribuição dos valores.

Em seguida, apliquei o `MinMaxScaler` para escalar os dados em um intervalo entre 0 e 1. Essa etapa é importante porque o KMeans é um algoritmo baseado em distância, então colunas com valores em escalas maiores podem influenciar mais o agrupamento do que deveriam. Ao escalar os dados, todas as variáveis passam a contribuir de maneira mais equilibrada para o cálculo das distâncias.

Após o escalonamento, converti os dados novamente para um `DataFrame` pandas, preservando os nomes das colunas originais para facilitar a análise. Também salvei o scaler treinado com `joblib`, permitindo reutilizar a mesma transformação em novos dados.

Com os dados escalados, executei novamente a avaliação para calcular inércia e Silhouette Score para diferentes quantidades de clusters, variando de 2 até 20. A partir dessa análise, utilizei o gráfico de silhueta e o método do cotovelo para apoiar a escolha da quantidade mais adequada de grupos.

Com base nos resultados observados, escolhi `k = 3` como configuração final. Por fim, treinei um novo modelo KMeans com 3 clusters usando os dados escalados, salvei o modelo em arquivo e gerei a base final com os agrupamentos.

Essa etapa reforçou a importância do pré-processamento em problemas de aprendizado não supervisionado, especialmente quando utilizamos algoritmos sensíveis à escala das variáveis, como o KMeans.
