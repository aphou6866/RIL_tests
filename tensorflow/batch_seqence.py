import tensorflow as tf
import pandas as pds
import os, sys, json


def generate_batch(n ,m):
    
    index=[]
    for i in range(n):
        index.append('idx%02d'%i)
    column=[]
    for i in range(m):
        column.append('col%02d'%i)
    df= pds.DataFrame(index=index, columns=column)
    c=0
    for i in range(n):
        for j in range(m):
            df.ix[i, j]= c
            c += 1
    #print(df)
    return df



def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])



def test_batch_to_sequence(df, n, m ,t):
    
    tf.reset_default_graph()
    input_data= tf.placeholder(tf.float32, [ None,m])
    x1=  input_data
    out= batch_to_seq( x1, t, m, flat=False)
    #out= tf.expand_dims( x1, axis=1,name='timely_input')
        
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    
    x1,out=sess.run( [x1, out],feed_dict={input_data:df} )
    print('X1.shape', x1.shape, 'out.len', len(out), out[0].shape)
    print(x1)
    print(out[0])
    print(out[1])

    
if __name__ == "__main__":
    
    argv=sys.argv
    
    n= int(argv[1])
    m= int(argv[2])
    t= int(argv[3])
    df= generate_batch(n ,m)
    test_batch_to_sequence(df, n,m,t)
