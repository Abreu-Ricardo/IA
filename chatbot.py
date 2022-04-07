# Aluno: Ricardo Abreu de Oliveira
# Professor: Edson Takashi

##### Passo 1. Conectar com o google drive para obter os dados 


# from google.colab import drive
# drive.mount('/content/drive')

# from google.colab import auth
# auth.authenticate_user()

# import gspread
# from google.auth import default
# creds, _ = default()

# gc = gspread.authorize(creds)

# worksheet = gc.open('garcom_diurno').sheet1

# # get_all_values gives a list of rows.
# rows = worksheet.get_all_values()
# print(rows)

##### Passo 1.1. Converter a tabela em um Data Frame em pandas

import pandas as pd
df=pd.DataFrame(rows[1:],columns=rows[0])


##### Passo 2.1. Divir(em xtrain_global e ytrain_global ) e transformar em lista para o treino
intencoes = ['fazer_pedido', 'fechar_conta', 'ver_cardapio']
xtrain_global = []
ytrain_global = []
for intencao in intencoes:
    lintencao = df[df['conjunto']=='Treino'][intencao].values.tolist()
    xtrain_global += lintencao
    ytrain_global += [intencao]*len(lintencao)

# Imprimir lista de teste
print(xtrain_global,ytrain_global)


##### Passo 2.2. Divir(em xtest ytest) e transformar em lista para o teste
xtest = []
ytest = []
for intencao in intencoes:
    lintencao = df[df['conjunto']=='Teste'][intencao].values.tolist()
    xtest += lintencao
    ytest += [intencao]*len(lintencao)

# Imprimir lista de teste
print(xtest,ytest)


##### Passo 3. Dividir os dados para validação e treino
import sklearn.model_selection as model_selection

xtrain, xval, ytrain, yval = model_selection.train_test_split(xtrain_global, ytrain_global, shuffle=True"Para embaralhar", stratify=ytrain_global)

np.unique(yval, return_counts=True)

##### Passo 4. Pré processamento, transformar as frases em sentence-embeddings

# No colab usar isso abaixo
#!pip install -U sentence-transformers


from sentence_transformers import SentenceTransformer

# Teste de tranformação de sentenças atribuindo significados

# converter = SentenceTransformer('multi-qa-distilbert-cos-v1')
# sent1=converter.encode('olah pessoal')
# print(sent1.shape)

# sentences = converter.encode(['bom dia pessoal','me veja um soba'])

# for sent in sentences:
#     print(np.linalg.norm(sent1-sent))


# Conversão do texto em xtrain, xval e xtest
xtrain_emb = converter.encode(xtrain)
xval_emb = converter.encode(xval)
xtest_emb = converter.encode(xtest)


##### Passo 4. Indução do modelo

import sklearn.model_selection as model_selection

model = neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform")

model.fit(xtrain_emb, ytrain)

KNClassifier()
