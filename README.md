# Clusterização: Lidando com dados sem rótulo

Este repositório reúne a evolução prática do curso de clusterização, organizado em **cinco atividades complementares**:

1. **Atividade 1 - Pipeline base com KMeans**
2. **Atividade 2 - Avaliação de métricas de clusterização**
3. **Atividade 3 - Otimização do resultado dos agrupamentos**
4. **Atividade 4 - Desvendando os agrupamentos**
5. **Atividade 5 - Aplicação web com Streamlit e deploy**

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

### Atividade 4 - Desvendando os agrupamentos

**Objetivo**: interpretar os clusters gerados pelo KMeans e identificar os perfis predominantes de cada grupo.

**Arquivos**:
- `atividade_desvendando_agrupamentos.ipynb`
- `atividade_desvendando_agrupamentos.py`

**Práticas aplicadas**:
- carregamento da base e preparação dos dados;
- One-Hot Encoding na coluna `sexo`;
- escalonamento com `MinMaxScaler`;
- treinamento do KMeans com 3 clusters;
- reversão da escala com `inverse_transform`;
- criação de `DataFrame` para análise em escala original;
- inclusão da coluna `cluster`;
- cálculo das médias por cluster;
- transposição e ordenação dos atributos mais relevantes;
- interpretação dos perfis de cada agrupamento;
- salvamento dos artefatos com `joblib`.

### Atividade 5 - Aplicação web com Streamlit e deploy

**Objetivo**: publicar a pipeline de clusterização em uma interface interativa, com upload de CSV, predição em lote e download do resultado.

**Arquivos**:
- `App.py`
- `app_core.py`
- `smoke_test_app.py`

**Práticas aplicadas**:
- separação entre camada de interface e camada de negócio;
- carregamento dos artefatos com cache em `st.cache_resource`;
- validação de estrutura e conteúdo do CSV;
- tratamento de exceções com mensagens claras para o usuário;
- exibição de preview, distribuição dos clusters e download do resultado;
- geração de CSV modelo para facilitar o uso;
- teste de fumaça local para validar o pipeline sem interface gráfica.

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

### Reversão da escala

Após o treinamento com dados escalados, o `inverse_transform` é usado para retornar os valores à escala original, facilitando a interpretação humana dos resultados.

### Análise por cluster

Com os rótulos dos clusters adicionados à base, `groupby('cluster').mean()` permite identificar quais atributos se destacam em cada grupo e apoiar decisões de negócio.

## Dependências

As dependências necessárias para todas as atividades estão em `requirements.txt`.

## Como executar

### Scripts Python

```bash
pip install -r requirements.txt
python atividade_clusterizacao.py
python atividade_metricas_clusterizacao.py
python atividade_otimizacao_clusterizacao.py
python atividade_desvendando_agrupamentos.py
python smoke_test_app.py
```

### Notebooks

```bash
pip install -r requirements.txt
jupyter notebook atividade_clusterizacao.ipynb
jupyter notebook atividade_metricas_clusterizacao.ipynb
jupyter notebook atividade_otimizacao_clusterizacao.ipynb
jupyter notebook atividade_desvendando_agrupamentos.ipynb
```

### Aplicação Streamlit

```bash
pip install -r requirements.txt
streamlit run App.py
```

Depois de iniciar, acesse `http://localhost:8501` no navegador.

## Documentação da aplicação

### Arquitetura

- `App.py`: interface Streamlit, widgets e experiência do usuário.
- `app_core.py`: pipeline de predição, validações e regras de negócio.
- `smoke_test_app.py`: teste rápido para validar artefatos e inferência.

Essa separação melhora manutenção, testes e evolução da aplicação.

### Fluxo de processamento

1. usuário envia um CSV na interface;
2. aplicação valida colunas obrigatórias e valores de `sexo`;
3. dados passam por One-Hot Encoding e escalonamento;
4. modelo `kmeans.pkl` gera os clusters;
5. resultado recebe `cluster` e `descricao_cluster`;
6. usuário visualiza dados e baixa o CSV final.

### Colunas esperadas no CSV de entrada

- `sexo`
- `idade`
- `numero_de_amigos`
- `basquete`
- `futebol_americano`
- `futebol`
- `softbol`
- `voleibol`
- `natacao`
- `animacao`
- `beisebol`
- `tenis`
- `esportes`
- `fofo`
- `danca`
- `banda`
- `marcha`
- `musica`
- `rock`
- `cabelo`
- `vestido`
- `shopping`
- `compras`
- `roupas`
- `nossa_marca`
- `marca_concorrente`
- `bebidas`

Valores aceitos para `sexo`: `F`, `M`, `NE`.

### Tratamento de erros (plus de robustez)

A aplicação implementa uma exceção específica (`AppDataError`) para cenários de entrada inválida, por exemplo:

- ausência de colunas obrigatórias;
- valores inválidos na coluna `sexo`;
- campos numéricos vazios ou com texto.

Com isso, o usuário recebe feedback objetivo e consegue corrigir o CSV rapidamente.

### Teste de fumaça

Para validar a pipeline sem abrir o navegador:

```bash
python smoke_test_app.py
```

Esse teste carrega os artefatos (`encoder.pkl`, `scaler.pkl`, `kmeans.pkl`), executa predição em amostra e confirma o formato da saída.

### Deploy no Streamlit Community Cloud

1. crie um repositório público no GitHub;
2. envie `App.py`, `app_core.py`, `requirements.txt` e os `.pkl`;
3. acesse `share.streamlit.io` e clique em **Create app**;
4. selecione repositório, branch `main` e arquivo `App.py`;
5. finalize em **Deploy** e compartilhe a URL `*.streamlit.app`.

## Artefatos gerados

Dependendo da atividade executada, os seguintes arquivos podem ser gerados ou atualizados:

- `encoder.pkl`
- `scaler.pkl`
- `modelo_kmeans.pkl`
- `kmeans.pkl`
- `dados_clusterizados.csv`
- `dados_clusterizados_escalados.csv`
- `dados_analise_clusterizados.csv`
- `medias_por_cluster.csv`

## Interpretação esperada dos grupos (Atividade 4)

Com base nas médias dos atributos por cluster, uma interpretação inicial dos perfis pode ser:

- **Grupo 0**: público jovem com maior interesse em moda, música e aparência;
- **Grupo 1**: público mais associado a esportes (especialmente futebol americano e basquete) e interesses culturais como banda e rock;
- **Grupo 2**: público mais equilibrado, com interesses em música, dança e moda.

## Texto para entrega - Atividade 4

Nesta atividade, dei continuidade ao fluxo de clusterização com foco na interpretação dos agrupamentos criados pelo modelo KMeans. Depois de preparar os dados, aplicar One-Hot Encoding, escalar as variáveis com MinMaxScaler e treinar o modelo com 3 clusters, o objetivo passou a ser entender o perfil de cada grupo formado.

Como o modelo foi treinado com os dados escalados, o primeiro passo foi reverter a escala utilizando o `inverse_transform` do scaler previamente treinado. Essa etapa é importante porque os dados escalados entre 0 e 1 são úteis para o modelo, mas não são tão intuitivos para análise humana. Ao retornar os valores para a escala original, fica mais fácil interpretar as características reais de cada agrupamento.

Em seguida, criei um novo `DataFrame` chamado `dados_analise`, contendo os dados revertidos para a escala original. Também adicionei a coluna `cluster`, utilizando os rótulos atribuídos pelo modelo KMeans. Com isso, foi possível agrupar os registros por cluster e calcular a média dos atributos de cada grupo.

Depois, transpus a tabela de médias para facilitar a comparação entre os clusters e ordenei os atributos de cada grupo em ordem decrescente. Essa análise permitiu identificar quais características eram mais fortes em cada agrupamento.

Com base nas médias analisadas, foi possível interpretar os três grupos principais: o grupo 0 apresentou um perfil mais jovem, com forte interesse em moda, música e aparência; o grupo 1 demonstrou maior associação com esportes, especialmente futebol americano, basquete e interesses culturais como banda e rock; já o grupo 2 apresentou um comportamento mais equilibrado, com interesses ligados a música, dança e moda.

Essa etapa foi importante porque mostrou que a clusterização não termina no treinamento do modelo. Depois que os grupos são criados, é necessário interpretá-los para transformar os resultados em informação útil e apoiar decisões, como estratégias de marketing, personalização de conteúdo ou segmentação de usuários.
