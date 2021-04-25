import copy
import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt



def valid(x,y):
    r = True
    if x < 0 : r = False
    if x > 2 : r = False
    if y < 0 : r = False
    if y > 2 : r = False
    return r



# s = [[4,1,3],[2,5,6],[0,7,8]]

# 4 1 3
# 2 5 6
# 0 7 8

def sons(s):
    r = []
    x = None
    y = None
    #localiza zero
    for i in range(len(s)):
        for j in range(len(s[i])):
            if s[i][j] == 0:
                x = i
                y = j
    # cima
    vx = x - 1
    vy = y
    if (valid(vx,vy)):
        ts = copy.deepcopy(s)
        t = ts[vx][vy]
        ts[vx][vy] = ts[x][y]
        ts[x][y] = t
        r.append(ts)
    # baixo
    vx = x + 1
    vy = y
    if (valid(vx,vy)):
        ts = copy.deepcopy(s)
        t = ts[vx][vy]
        ts[vx][vy] = ts[x][y]
        ts[x][y] = t
        r.append(ts)

    # direita
    vx = x 
    vy = y +1
    if (valid(vx,vy)):
        ts = copy.deepcopy(s)
        t = ts[vx][vy]
        ts[vx][vy] = ts[x][y]
        ts[x][y] = t
        r.append(ts)

    # esquerda
    vx = x 
    vy = y - 1
    if (valid(vx,vy)):
        ts = copy.deepcopy(s)
        t = ts[vx][vy]
        ts[vx][vy] = ts[x][y]
        ts[x][y] = t
        r.append(ts)

    return r

def printPuzzle(s):
    for v in s:
        print(v)

def son2str(s):
    s1 = s[0]+s[1]+s[2]
    return ''.join([str(v) for v in s1])


def bfs(start,goal):
    l = [start]
    fathers = dict()
    visited = [start]
    while (len(l)>0):
        father = l[0]
        del l[0]
        for son in sons(father):
            if son not in visited:
                visited.append(son)
                print(len(visited))
                fathers[son2str(son)] = father
                if son == goal:
                    res = []
                    node = son
                    while node != start:
                        res.append(node)
                        node = fathers[son2str(node)]
                    res.append(start)
                    res.reverse()
                    print(res)
                    return res
                else:
                    l.append(son)
    print("Sem Solucao")
##################################################################
def h2(a,b): # distancia de manhatan
    dist = 0
    tam = len(a)*len(a[0])
    v = [[] for i in range(tam)]
    for i in range(len(a)):
        for j in range(len(a[i])):
            v[a[i][j]].append((i,j))
            v[b[i][j]].append((i,j))
    for i in range(tam):
        dist += abs(v[i][0][0]-v[i][1][0]) + abs(v[i][0][1]-v[i][1][1])
    return dist


def busca_heuristica(start,goal,heuristica):
    h = []
    heappush(h,(heuristica(start,goal),start))
    fathers = dict()
    visited = [start]
    while (len(h)>0):
        (_,father) = heappop(h)
        for son in sons(father):
            if son not in visited:
                visited.append(son)
                visitados.append(visited)
                print(len(visited))
                fathers[son2str(son)] = father
                if son == goal:
                    res = []
                    node = son
                    while node != start:
                        res.append(node)
                        node = fathers[son2str(node)]
                    res.append(start)
                    res.reverse()
                    #print(res)
                    movimentos.append(res)
                    return res
                else:
                    heappush(h,(heuristica(son,goal),son))
    print("Sem Solucao")
######################################################

# Funcao de h1 implementada
def h1(start, goal):
    erros = int(0)

    for i in range(3):
        for j in range(3):
            if start[i][j] !=  goal[i][j]:
                erros += 1
        print()
    return erros

def resu(multivec):
    r = []
    for i in multivec:
        for j in i:
            for k in j:
                r.append(k)
    return r



#################################################33
# s = [[4,1,3],[2,5,6],[0,7,8]]

# # 4 1 3       1 2 3
# # 2 5 6   ->  4 5 6
# # 0 7 8       7 8 0
# resp = bfs(s,[[1,2,3],[4,5,6],[7,8,0]])
# for s in resp:
#     printPuzzle(s)
#     print()

##################################################
# start = [[4,1,3],[2,5,6],[0,7,8]]
# goal  = [[1,2,3],[4,5,6],[7,8,0]]
# # 4 1 3       1 2 3
# # 2 5 6   ->  4 5 6
# # 0 7 8       7 8 0
# # 1 +2 +0+1+0 +0+1+1+2 = 8
# h2(start,goal)


########################################################
#start1 = [[1,2,3], [4,6,8], [7,0,5]]
starth1_1 = [[1,2,3], [4,6,8], [7,0,5]] 
starth1_2 = [[1,2,3], [4,5,7], [0,8,6]] 
starth1_3 = [[1,2,3], [5,6,4], [7,0,8]]
starth1_4 = [[1,2,3], [4,7,6], [8,5,0]] 
starth1_5 = [[1,2,3], [4,7,0], [6,8,5]]


starth2_1 = [[1,2,3], [4,6,8], [7,0,5]] 
starth2_2 = [[1,2,3], [4,5,7], [0,8,6]] 
starth2_3 = [[1,2,3], [5,6,4], [7,0,8]]
starth2_4 = [[1,2,3], [4,7,6], [8,5,0]] 
starth2_5 = [[1,2,3], [4,7,0], [6,8,5]]

goal  = [[1,2,3],[4,5,6],[7,8,0]]
# 4 1 3       1 2 3
# 2 5 6   ->  4 5 6
# 0 7 8       7 8 0
# 1 +2 +0+1+0 +0+1+1+2 = 8

visitados = []
movimentos = []

resph1_1 = busca_heuristica(starth1_1,goal,h1)
resph1_2 = busca_heuristica(starth1_2,goal,h1)
resph1_3 = busca_heuristica(starth1_3,goal,h1)
resph1_4 = busca_heuristica(starth1_4,goal,h1)
resph1_5 = busca_heuristica(starth1_5,goal,h1)

resph2_1 = busca_heuristica(starth1_1,goal,h2)
resph2_2 = busca_heuristica(starth1_2,goal,h2)
resph2_3 = busca_heuristica(starth1_3,goal,h2)
resph2_4 = busca_heuristica(starth1_4,goal,h2)
resph2_5 = busca_heuristica(starth1_5,goal,h2)


r1 = resu(resph1_1)
r2 = resu(resph1_2)
r3 = resu(resph1_3)
r4 = resu(resph1_4)
r5 = resu(resph1_5)

a1 = resu(resph2_1)
a2 = resu(resph2_2)
a3 = resu(resph2_3)
a4 = resu(resph2_4)
a5 = resu(resph2_5)



movs = [5, 10, 15, 20, 25]
passos = [len(r1), len(r2), len(r3), len(r4), len(r5)]
pas =  [len(a1), len(a2), len(a3), len(a4), len(a5)]
plt.plot( movs, passos , label='h1')
plt.plot( movs, pas , label='h2')

plt.xlabel('movimentos')
plt.ylabel('nós visitados')
plt.legend()
plt.show()
