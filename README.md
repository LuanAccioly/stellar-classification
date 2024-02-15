## Classificação Estelar

**Algoritmos utilizados:**

- **Naive Bayes**
- **Random Forest**
- **Decision Tree**
- **KNN**

### Base de dados utilizada

[Link](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/)

> [!TIP]
> **Sobre a base**
> 
> Os dados consistem em 100.000 observações do espaço obtidas pelo SDSS (Sloan Digital Sky Survey). Cada observação é descrita por 17 colunas de características e 1 coluna de classe que a identifica como uma estrela, galáxia ou quasar.

**Algumas colunas presentes no dataset:**

- `alfa` - Ângulo de ascensão reta
- `delta` - Ângulo de declinação
- `u` - Filtro ultravioleta no sistema fotométrico
- `g` - Filtro verde no sistema fotométrico
- `r` - Filtro vermelho no sistema fotométrico
- `i` - Filtro infravermelho próximo no sistema fotométrico
- `z` - Filtro infravermelho no sistema fotométrico
- `redshift` - valor do redshift com base no aumento do comprimento de onda

#### Pré-Processamento utilizado

> [!NOTE]
> 1. **Ao menos um método de pré-processamento deve ser usado**, gerando uma nova base de dados.
> 1. O tipo de pré-processamento utilizado deve estar relacionado ao **contexto da aplicação**.
> 1. Remoção de vírgulas, espaços em branco, identificador dos padrões, etc. **não serão considerados pré-processamento válidos**.


##### MinMaxScaler: 

Utilizada para dimensionar (ou normalizar) os recursos em um intervalo especifico, geralmente entre `0 e 1` ou `-1 e 1`

**Funcionamento:**

$$
X_{\text{norm}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
$$

Onde:

- $X_{\text{norm}}$ é o valor normalizado do recurso
- $X$ é o valor original do recurso
- $X_{\text{min}}$ é o valor mínimo do recurso
- $X_{\text{max}}$ é o valor máximo do recurso

##### StandardScaler

Garante que os recursos tenham uma média zero e um desvio padrão comum

**Funcionamento:**

$$
X_{\text{new}} = \frac{X - \text{mean}(x)}{\text{std}(x)}
$$

Onde:

- $X$ é o valor original da característica
- $\text{mean}(x)$ é a média da característica
- $\text{std}(x)$ é o desvio padrão da característica

**Processando as bases:**

```python
# Base sem processamento
brute_X = brute_df.drop(columns=['obj_ID',
                                 'field_ID',
                                 'spec_obj_ID',
                                 'fiber_ID',
                                 'plate',
                                 'run_ID',
                                 'rerun_ID',
                                 'MJD',
                                 'class']).to_numpy()

# Classes
y = brute_df['class'].to_numpy()

# Base minMax. -1 : 1
minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
minmax_X = minmax_scaler.fit_transform(brute_X)

# Base stardard
standard_scaler = StandardScaler()
standard_X = standard_scaler.fit_transform(brute_X)
```

### Execução

>[!NOTE]
> **Descrição da atividade**
>
>1. **Deve-se executar o 10-*fold cross-validation* 5 vezes para cada base de dados**, com cada uma das cinco execuções partindo de uma distribuição aleatória dos dados entre cada *fold*, resultando em um total de 50 experimentos por base de dados (10 x 5).
> 1. Em cada um dos 50 experimentos, **os conjuntos de treinamento e teste devem ser mantido o mesmo para cada algoritmo a ser testado** (mesmo ponto de partida para cada modelo), de modo a obter-se uma avaliação justa dos resultados.
> 1. Ao menos **três algoritmos** devem ser testados e comparados:
>     1. **Árvore de Decisão;**
>      1. **Naïve Bayes;**
>     1. **K-Vizinhos Mais Próximos (K-NN) -** variando-se **3 vezes o número do parâmetro *k*;**
>      1. **Rede Neural Artificial treinada por *Backpropagation***.
>      1. **Outros Algoritmos de Aprendizagem Supervisionada (Classificadores) mediante validação prévia do Professor**.

**Definindo condições do projeto:**

```python
seeds = [2, 4, 8, 16, 32]
folds=10
algorithms = [rf, nb, knn, dt]
databases = [brute_X, minmax_X, standard_X]
```

- `seeds` - garantindo que os valores 'aleatórios' sejam os mesmos para cada uma das 5 interações
- `folds` - quantia de folds para o cross validation
- `algorithms` - Algoritmos utilizados
	- `RandomForest()`
	- `NaiveBayes()`
	- `KNN()`
	- `DecisionTree()`

**Executando:**

>[!WARNING]
> **Código resumido**
>
> Visando evitar excesso de informação no bloco de código, todas as linhas coletando dados (tempos de execução e tratamento dos scores) foram removidas deste resumo

```python
# Para cada base de dados (brute_X, processed_X)
for index, X in enumerate(databases):

  # Ao menos três algoritmos devem ser testados e comparados
  for algorithm in algorithms:

    # Cinco execuções partindo de uma distribuição aleatória
    # dos dados entre cada fold
    for seed in seeds:
      knn.n_neighbors = seed

      # Definindo a aleatoriedade dos folds
      kf = KFold(n_splits=folds,
                 shuffle=True,
                 random_state=seed)

      scores = cross_validate(algorithm,
                              X,
                              y,
                              cv=kf,
                              scoring=scoring_names)

```

### Resultados obtidos

---

#### Base bruta

**Tempo total**: 38 minutos

| Algoritmo      | Acurácia média | Precisão média | Recall médio | F1 médio | Tempo médio por seed | Tempo total |
|----------------|----------------|----------------|--------------|----------|----------------------|-------------|
| Random Forest  | 0.978829       | 0.978726       | 0.978829     | 0.978694 | 416.122              | 2080.612    |
| NaiveBayes     | 0.750596       | 0.794417       | 0.750596     | 0.694432 | 3.377                | 16.886      |
| Knn            | 0.816218       | 0.816819       | 0.816218     | 0.802933 | 14.091               | 70.453      |
| Decision Tree  | 0.964801       | 0.964881       | 0.964801     | 0.964835 | 23.168               | 115.840     |


#### Base MinMax

**Tempo total**:  35 minutos

| Algoritmo      | Acurácia média | Precisão média | Recall médio | F1 médio | Tempo médio por seed | Tempo total |
|----------------|----------------|----------------|--------------|----------|----------------------|-------------|
| Random Forest  | 0.977769       | 0.977657       | 0.977769     | 0.977631 | 379.574              | 1897.870    |
| NaiveBayes     | 0.763092       | 0.805737       | 0.763092     | 0.718866 | 3.573                | 17.865      |
| Knn            | 0.902516       | 0.905992       | 0.902516     | 0.901157 | 17.304               | 86.521      |
| Decision Tree  | 0.963368       | 0.963447       | 0.963368     | 0.963400 | 22.341               | 111.706     |

#### Base Standard

**Tempo total**: 38 minutos

| Algoritmo      | Acurácia média | Precisão média | Recall médio | F1 médio | Tempo médio por seed | Tempo total |
|----------------|----------------|----------------|--------------|----------|----------------------|-------------|
| Random Forest  | 0.978767       | 0.978663       | 0.978767     | 0.978632 | 414.971              | 2074.855    |
| NaiveBayes     | 0.763260       | 0.805852       | 0.763260     | 0.719109 | 3.366                | 16.828      |
| Knn            | 0.914789       | 0.916956       | 0.914789     | 0.913899 | 20.085               | 100.425     |
| Decision Tree  | 0.964659       | 0.964744       | 0.964659     | 0.964694 | 23.172               | 115.859     |

---
