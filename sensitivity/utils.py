import os, sys, json
import pandas as pds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import build_dense_net
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle 
from numpy import linalg as LA


def resample_dataframe(df,key):
    clsTypes={}
    for idx in df.index:
        cls= df[key][idx]
        if cls not in clsTypes:
            clsTypes[cls]={'counts':0, 'idx':0, 'indices':[] }
        clsTypes[cls]['counts'] += 1
        clsTypes[cls]['indices'].append(idx)
           
    maxCnt=0
    shuffled={}
    for cls in clsTypes:
        shuffled[cls]= shuffle(  clsTypes[cls]['indices'])
        if maxCnt < clsTypes[cls]['counts']:
            maxCnt= clsTypes[cls]['counts']
        print("%s: %d(%d) -->"%(cls, clsTypes[cls]['counts'], maxCnt))
    
    #sdf= pds.DataFrame(columns= df.columns)
    indices= []
    cc=0
    for cls in clsTypes:
        l= maxCnt -clsTypes[cls]['counts']
        
        for i in range(l):
            cc+=1
            j= clsTypes[cls]['idx']
            idx= shuffled[cls][j]
            clsTypes[cls]['idx']= (j+1)% clsTypes[cls]['counts']
            indices.append(idx)
            typ= df[key][idx]
            if cls != typ:
                print("Error")
                exit()

    #print(df.shape)
    sdf= df.filter(items=indices, axis=0).copy()
    #print(sdf[key])
    #print(sdf.shape)
    ndf= pds.concat([df, sdf], ignore_index=True)
    ndf.index=list( range(ndf.shape[0]))
    #print(ndf)
    #print(sdf.head(100))
    for cls in clsTypes:
        #print(type(cls))
        clsTypes[cls]['counts']=0
        
    for idx in ndf.index:
        cls= ndf[key][idx]
        clsTypes[cls]['counts'] += 1
           
    maxCnt=0
    for cls in clsTypes:
        if maxCnt < clsTypes[cls]['counts']:
            maxCnt= clsTypes[cls]['counts']
        print("%s: %d(%d) -->"%(cls, clsTypes[cls]['counts'], maxCnt))
    
    # Keep index in order while df[key] is out of order
    ndf= shuffle(ndf)
    ndf.reset_index(inplace=True)
    del ndf['index']
    return ndf


def  seperate_dataframe( df, key):
    
    clsTypes={}
    for idx in df.index:
        cls= df[key][idx]
        if cls not in clsTypes:
            clsTypes[cls]={'counts':0, 'idx':0, 'indices':[] }
        clsTypes[cls]['counts'] += 1
        clsTypes[cls]['indices'].append(idx)
           
    maxCnt=0
    shuffled={}
    for cls in clsTypes:
        shuffled[cls]= shuffle(  clsTypes[cls]['indices'])
        if maxCnt < clsTypes[cls]['counts']:
            maxCnt= clsTypes[cls]['counts']
        print("%s: %d(%d) -->"%(cls, clsTypes[cls]['counts'], maxCnt))

    print(df[df['rc']==0])

def split_train_test( df, key, pr, onehot):
    y= pds.DataFrame(index=df.index)
    X_train, X_test, y_train, y_test = train_test_split( df, y, test_size=pr)
    del df
    if onehot==True:
        y_test= pds.get_dummies( X_test[key])
    else:
        y_test= X_test[key]
    del X_test[key]
    
    X= shuffle(resample_dataframe(X_train, key))
    if onehot==True:
        y_train= pds.get_dummies( X[key])
    else:
        y_train= X[key]
    del X[key]
    print("split_train_test: ", X_train.shape, X.shape, X_test.shape, y_train.shape, y_test.shape)

    return (X, X_test, y_train, y_test)
