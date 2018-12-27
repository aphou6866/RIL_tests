import os, sys, json
import pandas as pds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import build_dense_net
from sklearn.metrics import accuracy_score
from numpy import linalg as LA
from utils import resample_dataframe, seperate_dataframe


np.random.seed(1)
tf.set_random_seed(1)

def generate_data_set(n):
    
    t= 3*np.pi 
    x=  np.linspace( -t, t, n, endpoint=False)
    y1= np.sin(x)
    y2= np.cos(x)
    #plt.plot(x, y1)
    #plt.plot(x, y2)
    
    ry= np.random.uniform(low= -2, high=2, size=n)
    rc1= np.where( y1< ry, 0, 1)
    rc2= np.where( ry< y2, 0, 1)
    rc= (rc1 + rc2)
    #plt.scatter(x, ry, c = rc)
    #plt.show()
    df=pds.DataFrame()
    df['x']= x
    df['y1']= y1
    df['y2']= y2
    df['ry']= ry
    df['rc']= rc
    print(max(rc), min(rc))
    #print(df)
    return df
    
    
def optimize_struct( n, m, lr):
    
    # loss function
    y_true= tf.placeholder(tf.float32, [None, m], 'y_true')
    X, y_train, params= build_dense_net( n, m, 2, 50, 'train_net')
    loss= tf.losses.softmax_cross_entropy( y_true, y_train)
    
    # sensitivity
    dX= tf.gradients( ys=loss, xs= X)
    ## Optimizer
    tf_steps = tf.Variable(0, trainable=False)
    inc_steps = tf_steps.assign(tf.add(tf_steps, 1))
    LR= tf.train.exponential_decay( lr, tf_steps, 100, .97, staircase=True)
    optimizer = tf.train.AdamOptimizer(LR)
    # Gradient Clipping
    dW = optimizer.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in dW if grad is not None]
    train = optimizer.apply_gradients(capped_gradients)
    #train= optimizer.minimize(loss)
    
    return {'X':X, 'y_train':y_train, 'y_true':y_true, 'params':params, 'loss':loss, 'dX':dX, 'dW':dW, 'inc_steps':inc_steps, 'train':train}


def train_data_altogether( df, sess, train_opt, valid_opt, m, epochs, batch_size):
    
    X=train_opt['X']
    y_true= train_opt['y_true']
    y_train= train_opt['y_train']
    loss= train_opt['loss']
    dX= train_opt['dX']
    dW= train_opt['dW']
    train= train_opt['train']
    inc_steps= train_opt['inc_steps']
    train_params= train_opt['params']
    valid_params= valid_opt['params']
    
    r=0.6
    ndf= resample_dataframe( df, 'rc')
    train_df= ndf[ ndf.index< ndf.shape[0]*r]
    valid_df= ndf[ ndf.shape[0]*r <= ndf.index]
    inValid= valid_df[['x', 'y1', 'y2', 'ry']].values
    outValid= np.eye(m)[valid_df['rc']]
    max_valid_accu= float('-inf')
    print('------>', max(ndf['rc']), max(train_df['rc']), max(valid_df['rc'])  )
    print( ndf[ndf['rc']==0 ].shape,  ndf[ndf['rc']==1 ].shape, ndf[ndf['rc']==2 ].shape)    
    print( train_df[train_df['rc']==0 ].shape,  train_df[train_df['rc']==1 ].shape, train_df[train_df['rc']==2 ].shape)
    
    for i in range(epochs):
        
        ndf= train_df.sample(frac=1)
        #ndf= df
        inTrain= ndf[['x', 'y1', 'y2', 'ry']].values
        outTrain= np.eye(m)[ndf['rc']]
        
        j =0 
        tol_train_loss= 0.
        tol_train_accu= 0.
        while j < train_df.shape[0]:
            
            x= inTrain[j:j+batch_size]
            Y= outTrain[j:j+batch_size]
            #if 0<j:
                #print( j, LA.norm( x- ox))
            ##print(Y)
            y, train_loss, gdx, _= sess.run( [ y_train, loss, dX, train], feed_dict={X: x, y_true: Y})
            train_accu= accuracy_score(np.argmax( y, axis=1), np.argmax(Y, axis=1))
            #print(j,x, gdx)
            
            #train_accu= np.sum( np.where( np.argmax( y, axis=1)==np.argmax(Y, axis=1), 1, 0) )
            nx= LA.norm(gdx[0], axis=1)
            if j==0:
                normDX= nx
            else:
                normDX = np.append( normDX, nx)
            #print(normDX.shape )
            #print(np.argmax( y, axis=1), np.argmax(Y, axis=1), tol_train_accu, train_accu)
            
            
            tol_train_loss += train_loss
            tol_train_accu += train_accu
            j = j+batch_size
            #sess.run([inc_steps])
            #print(j, df.shape[0])
        print('Delta X:',normDX)
        tol_train_loss= (tol_train_loss*batch_size) / train_df.shape[0]
        tol_train_accu= (tol_train_accu*batch_size) / train_df.shape[0]
        #tol_train_accu= (tol_train_accu*batch_size) / df.shape[0]
        
        y, valid_loss, gdx= sess.run( [ y_train, loss, dX], feed_dict={X: inValid, y_true: outValid})
        valid_accu= accuracy_score(np.argmax( y, axis=1), np.argmax(outValid, axis=1))
        #meraged_accu= 0.5 * (valid_accu + tol_train_accu)
        #if max_valid_accu < meraged_accu:
            #max_valid_accu = meraged_accu
            #sess.run([tf.assign(v, t) for v, t in zip( valid_params, train_params)])
        if max_valid_accu < valid_accu:
            max_valid_accu = valid_accu
            sess.run([tf.assign( v, t) for v, t in zip( valid_params, train_params)])
        else:
            sess.run([tf.assign( t, v) for t, v in zip( train_params, valid_params )])
        
        print( j, "Epoch-%d"%i, "Train: loss-",tol_train_loss, "accuracy-", tol_train_accu, " Valid: loss-",valid_loss, "accuracy- %f(%f)"%(valid_accu, max_valid_accu))
        
    sess.run([tf.assign(t, v) for t, v in zip( train_params, valid_params )])
    


def reference_dataframe( df, sess, opt, m, batch_size):
    
    X=opt['X']
    y_train= opt['y_train']
    y_true= opt['y_true']
    loss= opt['loss']
    dX= opt['dX']
    dW= opt['dW']
    train= opt['train']
    
    inData= df[['x', 'y1', 'y2', 'ry']].values
    outData= np.eye(m)[df['rc']]
    j= 0
    tol_train_loss= 0.
    tol_train_accu= 0.
    while j < df.shape[0]:
        
        x= inData[j:j+batch_size]
        Y= outData[j:j+batch_size]
        
        y, train_loss, gdx= sess.run( [ y_train, loss, dX], feed_dict={X: x, y_true: Y})
        a= np.argmax( y, axis=1)
        b= np.argmax( Y, axis=1)
        train_accu= accuracy_score( a, b)
    
        if j==0:
            rc1= a
        else:
            rc1= np.append(rc1,  a)
        if j==0:
            rc2= b
        else:
            rc2= np.append(rc2,  b)
            
        j = j+batch_size
        tol_train_loss += train_loss
        tol_train_accu += train_accu
    
    tol_train_loss= (tol_train_loss*batch_size) / df.shape[0]
    tol_train_accu= (tol_train_accu*batch_size) / df.shape[0]
    print("Reference loss:",tol_train_loss, "accuracy:", tol_train_accu)
    print( max(rc1), max(rc2), min(rc1), min(rc2))
    
    tdata= np.transpose(inData)
    plt.figure(1)
    plt.plot(tdata[0], tdata[1])
    plt.plot(tdata[0], tdata[2])
    plt.scatter(tdata[0], tdata[3], c = rc1)
    
    plt.figure(2)
    plt.plot(tdata[0], tdata[1])
    plt.plot(tdata[0], tdata[2])
    plt.scatter(tdata[0], tdata[3], c = rc2)
    
    plt.show()
    
    
def test_stability( df, n, m, lr, epochs, batch_size):
    
    print(" n,m, lr===>",n,m, lr)
    sess = tf.Session()
    
    train_opt=optimize_struct( n, m, lr)
    X, y, params= build_dense_net( n, m, 2, 50, 'valid_net')
    valid_opt= {'X':X, 'y_train':y, 'params':params}
    
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter( './logs', sess.graph)
    
    train_data_altogether( df, sess, train_opt, valid_opt, m, epochs, batch_size)
    #train_data_by_class( df, sess, train_opt, valid_opt, m, epochs, batch_size)
    reference_dataframe( df, sess, train_opt, m, batch_size)
    #print(inData)
    #print(outData)
    
            
            

if __name__ == "__main__":
    
    argv=sys.argv
    samples=  int(argv[1])
    epochs=  int(argv[2])
   
    df= generate_data_set(samples)
    test_stability( df, len(df.columns)-1, max(df['rc']) +1, 0.001, epochs, 50)
    
    
    
