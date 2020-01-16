# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:56:22 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float64')


def cross_entropy(x, pred):
    ce = tf.reduce_mean(-tf.reduce_sum(x*tf.math.log(tf.clip_by_value(pred, 1e-10, 1.0))
            + (1-x)*tf.math.log(tf.clip_by_value(1-pred,1e-10,1.0)), 1))
    return ce



def step_lr(epochset, lrset):
    Lr = np.array(range(epochset[-1]), dtype = np.float64)
    for i, lr in enumerate(lrset[::-1]):
        Lr[:epochset[len(epochset) - 1 - i]] = lr
    return Lr
    


def cycle_fn(iteration, base_lr, max_lr, stepsize):
    cycle = np.floor(1+iteration/(2*stepsize))
    x = np.abs(iteration/stepsize - 2*cycle +1)
    lr = base_lr + (max_lr - base_lr)*np.maximum(0, (1-x))
    return np.float64(lr)


def cycle_lr(base_lr, max_lr, iter_in_batch, epoch_for_cycle, ratio, total_epochs):
    iteration = 0;
    Lr = [];
    stepsize = (iter_in_batch*epoch_for_cycle)/2.
    for i in range(total_epochs):
        for j in range(iter_in_batch):
            Lr.append(cycle_fn(iteration, base_lr = base_lr, 
                            max_lr = max_lr, stepsize = stepsize))
            iteration+=1
    final_iter = np.int((total_epochs/epoch_for_cycle)*stepsize*2*ratio)
    Lr = np.array(Lr)
    Lr[final_iter:] = base_lr
    return Lr

def c_max(inputs, a):
    return (inputs + a + tf.abs(inputs - a))/2

def c_min(inputs, a):
    return (inputs + a - tf.abs(inputs - a))/2


def c_sign(inputs):
    return c_min(c_max(inputs*10, 0), 1)
    

def c_ball(inputs, a):
    x = -tf.nn.relu(inputs) + a
    x = c_sign(x)
    x = tf.multiply(inputs, x)
    return tf.nn.relu(x)


def reduce_var(inputs):
    m = tf.reduce_mean(inputs, axis = 0)
    square = tf.reduce_sum((inputs - m) * (inputs - m), axis = -1)
    return tf.reduce_mean(square)



class MADE(layers.Layer):
    def __init__(self, n_latent = 32, n_hidden = 32):
        super(MADE, self).__init__()
        # w
        w_init = tf.initializers.GlorotUniform()
        self.w = tf.Variable(initial_value = w_init(shape = (n_latent + n_hidden, n_hidden),
                                                    dtype = 'float64'), trainable = True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value = b_init(shape = (n_hidden,), dtype = 'float64'),
                             trainable = True)
        
        # vm
        vm_init = tf.initializers.GlorotUniform()        
        self.vm = tf.Variable(initial_value = vm_init(shape = (n_hidden, n_latent),
                                                    dtype = 'float64'), trainable = True)
        cm_init = tf.zeros_initializer()
        self.cm = tf.Variable(initial_value = cm_init(shape = (n_latent,), dtype = 'float64'),
                             trainable = True)

        # vs
        vs_init = tf.initializers.GlorotUniform()        
        self.vs = tf.Variable(initial_value = vs_init(shape = (n_hidden, n_latent),
                                                    dtype = 'float64'), trainable = True)
        cs_init = tf.ones_initializer()
        self.cs = tf.Variable(initial_value = cs_init(shape = (n_latent,), dtype = 'float64') * 2.0,
                             trainable = True)
        
        self.W_mask, self.V_mask = self.Generate_mask(n_latent, n_hidden)

    def Generate_mask(self, n_latent, n_hidden):
        max_masks = np.random.randint(low = 1, high = n_latent, size = n_hidden)
        W_mask = np.fromfunction(lambda d, k: max_masks[k] >= d+1, 
                                 (n_latent + n_hidden, n_hidden),
                                 dtype = int).astype(np.float64)
        W_mask = tf.Variable(W_mask, trainable=False, dtype = 'float64')

        V_mask = np.fromfunction(lambda k, d: d + 1 > max_masks[k], 
                                 (n_hidden, n_latent),
                                 dtype=int).astype(np.float64)
        V_mask = tf.Variable(V_mask, trainable=False, dtype = 'float64')

        return W_mask, V_mask

        
    def call(self, z, h):
        W = tf.multiply(self.w, self.W_mask)
        V_m = tf.multiply(self.vm, self.V_mask)
        V_s = tf.multiply(self.vs, self.V_mask)
           
        x = tf.matmul(tf.concat([z, h], axis = 1), W) + self.b
        x = tf.nn.relu(x)
        
        m = tf.matmul(x, V_m) + self.cm
        s = tf.matmul(x, V_s) + self.cs
        return m, s
  

class DiagonalGaussian(object):
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        
    def log_probability(self, x):
        return -0.5*tf.reduce_sum(np.log(2.0*np.pi) + self.logvar + ((x-self.mu)**2)
                                  /tf.exp(self.logvar), axis = 1)
        
    def sample(self):
        eps = tf.random.normal(tf.shape(self.mu), 0, 1, dtype = tf.float64)
        return self.mu + tf.exp(0.5 * self.logvar)*eps
    
    def repeat(self, n):
        mu = tf.reshape(tf.tile(tf.expand_dims(self.mu, 1), [1, n, 1]), shape = (-1, self.mu.shape[-1]))
        var = tf.reshape(tf.tile(tf.expand_dims(self.logvar, 1), [1, n, 1]), shape = (-1, self.logvar.shape[-1]))
        return DiagonalGaussian(mu, var)
    
    def kl_div(p, q):
        return 0.5*tf.reduce_sum(q.logvar - p.logvar - 1.0 + (tf.exp(p.logvar) + (p.mu - q.mu)**2)/(tf.exp(q.logvar)), axis = 1)

        

class Gaussian(object):
    def __init__(self, mu, precision):
        self.mu = mu
        self.precision = precision #(batch_size, lat_dim, lat_dim)
        self.L = tf.linalg.cholesky(tf.linalg.inv(precision))
        self.dim = tf.cast(tf.shape(self.mu)[1], tf.float64)

    def log_probability(self, x):
        return -0.5 * (self.dim * np.log(2.0*np.pi)
                       + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)), axis = 1)
                       +tf.reduce_sum(tf.reduce_sum(tf.matmul(tf.matmul(tf.expand_dims((x - self.mu), 1), self.precision),
                                  (tf.expand_dims((x - self.mu), -1))), axis = 2), axis =1))
                         
    def sample(self):
        eps = tf.random_normal(tf.shape(self.mu), 0, 1, dtype = tf.float64)
        return self.mu + tf.squeeze(tf.matmul(self.L, tf.expand_dims(eps, -1)), -1)
        


class Transformation(object):
    def __init__(self, ball):
        self.ball = ball
          
    def interpolate(self, radi):
        rnd_vec = tf.random.uniform(shape=(tf.shape(self.ball)), minval = -1, maxval = 1, dtype = tf.float64)
        unif = tf.random.uniform(shape=(tf.shape(self.ball)[0],), minval = 0, maxval = 1, dtype = tf.float64)
        scale_f = tf.expand_dims(tf.norm(rnd_vec, axis=1)/unif, axis=1)
        rnd_vec = rnd_vec/scale_f
        return self.ball + radi*rnd_vec

    def diff(self, x):
        return tf.reduce_sum((x-self.ball)**2, axis = 1)
