# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:49:56 2020

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
from models import FCN32s, FCN16s, FCN8s
from networks import step_lr
import tensorflow as tf
import numpy as np
import os
from bird import Bird
import argparse
from collections import OrderedDict
np.random.seed(0)
tf.random.set_seed(0)
tf.keras.backend.set_floatx('float64')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 'FCN32s')
    parser.add_argument('--datasets', type = str, default = 'Bird')
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--ep_set', type = list, default = [15, 25, 50])
    parser.add_argument('--batch_size', type = int, default = 10)
    parser.add_argument('--lr_set', type = list, default = [0.0005, 0.0001, 0.00005])
    parser.add_argument('--dim', type = int, default = 128)
    
    args = parser.parse_args()
    
    config = OrderedDict([
            ('model_name', args.model_name),
            ('datasets', args.datasets),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lr_set', args.lr_set),
            ('ep_set', args.ep_set),
            ('dim', args.dim),])
    
    return config
    
config = parse_args()

bird = Bird('Unet')

n_samples = bird.num_train_examples

dim = config['dim']


if config['model_name'] == 'FCN32s':
    print('Run FCN32s')
    model = FCN32s(256).build()    
elif config['model_name'] == 'FCN16s':
    print('Run FCN16s')
    model = FCN16s(256).build()
elif config['model_name'] == 'FCN8s':
    print('Run FCN8s')
    model = FCN8s(256).build()    

mother_folder = os.path.join('Unet',config['model_name'])
try:
    os.mkdir(mother_folder)
except OSError:
    pass    

folder_name = os.path.join(mother_folder, config['model_name']+'_'+config['datasets']+'_'+str(config['dim']))
try:
    os.mkdir(folder_name)
except OSError:
    pass    

optimizer = tf.keras.optimizers.Adam(lr=config['lr_set'][0])
loss_object = tf.nn.sparse_softmax_cross_entropy_with_logits

train_loss = tf.keras.metrics.Mean(name='train_loss')

summary_writer = tf.summary.create_file_writer(folder_name)

Lr = step_lr(config['ep_set'], config['lr_set'])

@tf.function
def train_step(X, Y, i):
    tf.keras.backend.set_value(optimizer.lr, i)
    with tf.GradientTape() as tape:
        pred, annot = model(X)
        loss = loss_object(Y, annot)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


### run ###
iter_per_epoch = int(n_samples/config['batch_size']) 
def runs(log_freq = 1):
    for epoch in range(config['epochs']):
        for iter_in_epoch in range(iter_per_epoch):
            images, annots, labels = bird.next_train_batch(config['batch_size'])
            train_step(tf.cast(images, tf.float64), tf.cast(annots, tf.int64), Lr[epoch])
            if tf.equal(optimizer.iterations % log_freq, 0):
                tf.summary.scalar('loss', train_loss.result(), step=optimizer.iterations)
                
        template = 'epoch: {}, completed out of: {}, train_loss: {}'
        print(template.format(epoch+1,
                              config['epochs'],
                                 train_loss.result()))   
        
             
train_summary_writer = tf.summary.create_file_writer(folder_name+'/train')

runs()

model_folder = os.path.join(folder_name, 'model')

tf.keras.models.save_model(model, model_folder)
#
#
#
