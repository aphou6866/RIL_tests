#!/usr/bin/env python3

import numpy as np
import json, os, sys
from random import randint, seed

def create_random_path( n, m, k, positive, shortest):
    print(shortest)
    seed(10)
    
    if shortest== True:
        mx= 1000000
    elif positive==True:
        mx=0
    else:
        mx= -1000000
    graph= [[ mx for x in range(n)] for y in range(n)] 
        
    i=0
    while i < (n-1):
        if positive==True:
            c= randint(1,100)
        else:
            c= 100- randint(1,200)
            
        l= randint(1,5)
        #print(l)
        j= l+i
        if n<=j:
            j=n-1
        graph[i][j]=c
        i= j
    if i< (n-1):
        c= randint(1,100)
        graph[i][n-1]= c
        
    for i in range(k):
        if shortest==True:
            p= randint(0,n-2)
            q= randint(0,n-1)
        else:
            p= randint(0,n-2)
            q= randint(p+1,n-1)
                
            
        c= randint(1, 100)
        graph[p][q]= c
    
    edges= get_edges(graph, mx)
        
    for line in graph:
        print(line)
    #for e in edges:
        #print(e)
    return (graph, edges, mx)


def get_edges( graph, mx):
    n= len(graph)
    edges= []
    for i in range(n):
        for j in range(n):
            dist= graph[i][j]
            if dist!=mx:
                edges.append( {'src':i, 'dst':j, 'weight':dist} )
    return edges


def algo_Dijkstra( graph, src, mx):
    
    n= len(graph)
    dist= [mx]*n
    prev= [-1]*n
    Q=  list(range(n))
    dist[src]= 0

    while 0<len(Q):
        u=-1
        m=mx-1
        for i in Q:
            if dist[i]< m:
                m= dist[i]
                u=i
        #print(dist, Q)
        #print(u)
        if u in Q:
            Q.remove(u)
        else:
            break
        for v in range(n):
            alt= dist[u]+ graph[u][v]
            if alt < dist[v]:
                dist[v]= alt
                prev[v]= u
        
        #break
    #return dist, prev
    
    print('Distances:',dist)
    print('Previous:',prev)
    i=n-1
    path=[]
    dist=0
    while 0<i:
        j= prev[i]
        w= graph[j][i]
        dist += w
        path.append({'src':j,'dst':i, 'dist':w})
        i= j
    #for line in graph:
        #print(line)
    for p in path:
        print( p)
    
    
def algo_Bellman( graph, edges, src, mx, shortest):
    
    n= len(graph)
    distance= [mx]*n
    predecessor= [-1]*n
    Q=  list(range(n))
    distance[src]= 0
    
    # relax edges repeatedly
    for i in range(n-1):
        for e in edges:
            u= e['src']
            v= e['dst']
            w= e['weight']
            if shortest== True and distance[u] + w < distance[v]:
                distance[v]= distance[u] + w
                predecessor[v]= u
            elif shortest== False and distance[v] < distance[u] + w :
                distance[v]= distance[u] + w
                predecessor[v]= u
            
    
    # check for negative-weight cycles
    for e in edges:
        u= e['src']
        v= e['dst']
        w= e['weight']
        if shortest== True and distance[u] + w < distance[v]:
            print(  "Graph contains a negative-weight cycle")
        elif shortest== True and distance[v] < distance[u] + w:
            print(  "Graph contains a negative-weight cycle")
    
    print('Distances:',distance)
    print('Predecessor:',predecessor)
    i=n-1
    path=[]
    dist=0
    while 0<i:
        j= predecessor[i]
        w= graph[j][i]
        dist += w
        path.append({'src':j,'dst':i, 'weight':w })
        i= j
    #for line in graph:
        #print(line)
    for p in path:
        print( p)
    print('Total distance:', dist)
    

# ./shortest_path.py 10 2 15 long

if __name__ == "__main__":
    
    
    argv= sys.argv
    n= int(argv[1])
    m= int(argv[2])
    k= int(argv[3])
    if argv[4]=='short':
        shortest= True
    else:
        shortest= False
    (graph, edges, mx) = create_random_path( n, m, k,False, shortest)
    print("mx:",mx)
    #print("\n------------Dijkstra Algorithm------------------\n")
    #algo_Dijkstra(graph, 0, mx  )
    print("\n------------Bellman Algorithm------------------\n")
    algo_Bellman( graph, edges, 0, mx, shortest)
