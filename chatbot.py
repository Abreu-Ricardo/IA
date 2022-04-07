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


########### FAZER A DOCUMENTACAO AIUNDA

"""# Ajuste de parâmetros"""

model.predict([converter.encode('olah pessoal')])

"""# Avaliação do Modelo"""

import sklearn.metrics as metrics

pred = model.predict(xtest_emb)

"""# Teste"""

model.predict([converter.encode('gostaria de pedir um soba')])



"""#Modelo final"""

model.fit(converter.encode(xtrain_global),ytrain_global)

from joblib import dump, load

dump(model, 'garcom.joblib')

modelo_final = load('garcom.joblib')

modelo_final.predict([converter.encode('fecha a conta')])



"""# Construção de um chatbot"""

def convert_num(n):
    valores = {'um':1,
               'uma':1,
               'dois':2,
               'duas':2,
               'tres':3,
               'quatro':4,
               'cinco':5,
               'seis':6,
               'sete':7,
               'oito':8,
               'nove':9,
               'dez':10}
    ret = 0
    if n.isnumeric():
        ret = int(n)
    else:
        if n in valores.keys():
            ret = valores[n]

    return ret

!pip install unidecode

from nltk.tokenize import TweetTokenizer
from unidecode import unidecode
tknzr = TweetTokenizer()

lista_entidades = [
'item:coca,cocas,coca-cola,guarana,agua,guaranas',
'item:soba,espeto,sobas,espetos',
'num:1,2,3,4,5,6,7,8,9,10',
'num:um,dois,tres,quatro,cinco,seis,sete,oito,nove,dez,uma,duas'
]
entidades = dict()
def load_entidades(lista_entidades):
            for line in lista_entidades:
                entidade,valores = line.split(':')
                str_valores = valores[:]
                valores = str_valores.split(',')
                for valor in valores:
                    if valor not in entidades.keys():
                        entidades[valor] = entidade
load_entidades(lista_entidades)
def find_entidades(texto):
    ret = dict()
    for token in tknzr.tokenize(texto):
        token = token.lower()
        token = unidecode(token)
        if token in entidades.keys():
            ent = entidades[token]
            if ent not in ret.keys():
                ret[ent] = [token]
            else:
                ret[ent] += [token]
    return ret

def convert_num(n):
    valores = {'um':1,
               'uma':1,
               'dois':2,
               'duas':2,
               'tres':3,
               'quatro':4,
               'cinco':5,
               'seis':6,
               'sete':7,
               'oito':8,
               'nove':9,
               'dez':10}
    ret = 0
    if n.isnumeric():
        ret = int(n)
    else:
        if n in valores.keys():
            ret = valores[n]

    return ret

print(find_entidades('me ve um sobá ai'))

valor_cardapio = {
    'soba':20.0,
    'espeto':20.0,
    'coca':5.0,
    'guarana':5.0
}

def str_menu(h):
    rstr = ''
    for item in h:
        rstr += f"{item:<10}  {h[item]:>5}\n"
    return rstr

print(str_menu(valor_cardapio))



sair = False
#intencoes = ['fazer_pedido', 'fechar_conta', 'ver_cardapio']
print("Ola, seja bem vindo a sobaria da FACOM, o que gostaria de pedir?")
pedidos = []
while(not sair):
    cliente = input("<")
    msg = ''
    pred = modelo_final.predict([converter.encode(cliente)])[0]
    if pred == 'ver_cardapio':
        msg = str_menu(valor_cardapio)
    elif pred == 'fazer_pedido':
        ent = find_entidades(cliente)
        print(ent)
        joined = [[x,y] for x,y in zip(ent['num'],ent['item'])]
        pedidos.append(joined)
        msg += 'ok '
        for n,item in joined[:-1]:
            msg += ',%s %s '%(n,item)
        msg += ('e %s %s '%(joined[-1][0],joined[-1][1]))        
        msg += 'saindo\n'
    elif pred == 'fechar_conta':
        msg += 'Ok, foi pedido:\n'
        total = 0
        for joined in pedidos:
            for n,item in joined:
                nvalor = convert_num(n)
                if item in valor_cardapio.keys():
                    vitem = valor_cardapio[item]
                total += vitem*nvalor
                msg += '%s %s %04.2f\n'%(n,item,vitem*nvalor)
            
        msg += ('Total foi R$ %4.2f\n'%total)
    elif pred == 'bye':
        msg = 'Bye, volte sempre\n'
        sair = True
    else:
        msg = 'Não, entendi. \n'+str_menu(valor_cardapio)
    print(msg)

