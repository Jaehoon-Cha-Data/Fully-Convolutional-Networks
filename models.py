# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 23:03:41 2020

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout 
from tensorflow.keras.models import Model


class FCN32s(Model):
    def __init__(self, n_classes, input_shape):
        super(FCN32s, self).__init__()
        self.n_classes = n_classes
        
        self.vgg16  = tf.keras.applications.VGG16(include_top=False,weights='imagenet')
        self.vgg16.trainable = True
        
        self.b1_conv1 = self.vgg16.get_layer(self.vgg16.layers[1].name)
        self.b1_conv2 = self.vgg16.get_layer(self.vgg16.layers[2].name)
        self.b1_pool1 = self.vgg16.get_layer(self.vgg16.layers[3].name)
        
        self.b2_conv1 = self.vgg16.get_layer(self.vgg16.layers[4].name)
        self.b2_conv2 = self.vgg16.get_layer(self.vgg16.layers[5].name)
        self.b2_pool1 = self.vgg16.get_layer(self.vgg16.layers[6].name)        

        self.b3_conv1 = self.vgg16.get_layer(self.vgg16.layers[7].name)
        self.b3_conv2 = self.vgg16.get_layer(self.vgg16.layers[8].name)
        self.b3_conv3 = self.vgg16.get_layer(self.vgg16.layers[9].name)
        self.b3_pool1 = self.vgg16.get_layer(self.vgg16.layers[10].name)  
        
        self.b4_conv1 = self.vgg16.get_layer(self.vgg16.layers[11].name)
        self.b4_conv2 = self.vgg16.get_layer(self.vgg16.layers[12].name)
        self.b4_conv3 = self.vgg16.get_layer(self.vgg16.layers[13].name)
        self.b4_pool1 = self.vgg16.get_layer(self.vgg16.layers[14].name)          

        self.b5_conv1 = self.vgg16.get_layer(self.vgg16.layers[15].name)
        self.b5_conv2 = self.vgg16.get_layer(self.vgg16.layers[16].name)
        self.b5_conv3 = self.vgg16.get_layer(self.vgg16.layers[17].name)
        self.b5_pool1 = self.vgg16.get_layer(self.vgg16.layers[18].name)  

        self.b6_conv1 = Conv2D(4096, 7, 1, padding = 'same',
                                    activation = 'relu',
                                    name='block6_conv1')     
        self.b6_dropout1 = Dropout(0.5)
        
        self.b7_conv1 = Conv2D(4096, 1, 1, padding = 'same',
                                   activation = 'relu',
                                    name='block7_conv1')
        self.b7_dropout1 = Dropout(0.5)

        self.s32_conv1 = Conv2D(self.n_classes, 1, 1, padding = 'same',
                                name = 's32_conv1') #200=n_classes

        self.d32_conv1 = Conv2DTranspose(self.n_classes, 64, 32, padding = 'same',
                                         name = 'd32_conv1')

    def call(self, X):
        x = self.b1_conv1(X[0])
        x = self.b1_conv2(x)
        b1_out = self.b1_pool1(x)
        
        x = self.b2_conv1(b1_out)
        x = self.b2_conv2(x)
        b2_out = self.b2_pool1(x)
        
        x = self.b3_conv1(b2_out)
        x = self.b3_conv2(x)
        x = self.b3_conv3(x)
        b3_out = self.b3_pool1(x)
        
        x = self.b4_conv1(b3_out)
        x = self.b4_conv2(x)
        x = self.b4_conv3(x)
        b4_out = self.b4_pool1(x)
        
        x = self.b5_conv1(b4_out)
        x = self.b5_conv2(x)
        x = self.b5_conv3(x)
        b5_out = self.b5_pool1(x)
        
        x = self.b6_conv1(b5_out)
        b6_out = self.b6_dropout1(x)
        x = self.b7_conv1(b6_out)
        b7_out = self.b7_dropout1(x)
        
        score32 = self.s32_conv1(b7_out)
        self.d32conv = self.d32_conv1(score32)
        
        self.pred = tf.argmax(self.d32conv, axis=-1, name="prediction")
        
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = X[1],
                                                                logits = self.d32conv)
        return self.pred, self.d32conv, self.loss


class FCN16s(FCN32s):
    def __init__(self, n_classes, input_shape):
        super(FCN16s, self).__init__(n_classes, input_shape)
        self.s32_conv1 = Conv2D(self.n_classes, 1, 1, padding = 'same',
                                name = 's32_conv1') #200=n_classes
        
        self.s16_conv1 = Conv2D(self.n_classes, 1, 1, padding = 'same',
                                name = 's16_conv1')

        self.d32_conv1 = Conv2DTranspose(self.n_classes, 4, 2, padding = 'same',
                                         name = 'd32_conv1')
        
        self.d16_conv1 = Conv2DTranspose(self.n_classes, 16, 16, padding = 'same',
                                         name = 'd16_conv1')
    def call(self, X):
        x = self.b1_conv1(X[0])
        x = self.b1_conv2(x)
        b1_out = self.b1_pool1(x)
        
        x = self.b2_conv1(b1_out)
        x = self.b2_conv2(x)
        b2_out = self.b2_pool1(x)
        
        x = self.b3_conv1(b2_out)
        x = self.b3_conv2(x)
        x = self.b3_conv3(x)
        b3_out = self.b3_pool1(x)
        
        x = self.b4_conv1(b3_out)
        x = self.b4_conv2(x)
        x = self.b4_conv3(x)
        b4_out = self.b4_pool1(x)
        
        x = self.b5_conv1(b4_out)
        x = self.b5_conv2(x)
        x = self.b5_conv3(x)
        b5_out = self.b5_pool1(x)
        
        x = self.b6_conv1(b5_out)
        b6_out = self.b6_dropout1(x)
        x = self.b7_conv1(b6_out)
        b7_out = self.b7_dropout1(x)
        
        score32 = self.s32_conv1(b7_out)
        
        score16 = self.s16_conv1(b4_out)
        
        self.d32conv = self.d32_conv1(score32)
        
        self.d16conv = self.d16_conv1(self.d32conv + score16)
        
        self.pred = tf.argmax(self.d16conv, axis=-1, name="prediction")
        
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = X[1],
                                                                logits = self.d16conv)
        return self.pred, self.d16conv, self.loss


class FCN8s(FCN32s):
    def __init__(self, n_classes, input_shape):
        super(FCN8s, self).__init__(n_classes, input_shape)
        self.s32_conv1 = Conv2D(self.n_classes, 1, 1, padding = 'same',
                                name = 's32_conv1') #200=n_classes
        
        self.s16_conv1 = Conv2D(self.n_classes, 1, 1, padding = 'same',
                                name = 's16_conv1')

        self.s8_conv1 = Conv2D(self.n_classes, 1, 1, padding = 'same',
                                name = 's8_conv1')

        self.d32_conv1 = Conv2DTranspose(self.n_classes, 4, 2, padding = 'same',
                                         name = 'd32_conv1')
        
        self.d16_conv1 = Conv2DTranspose(self.n_classes, 4, 2, padding = 'same',
                                         name = 'd16_conv1')

        self.d8_conv1 = Conv2DTranspose(self.n_classes, 4, 8, padding = 'same',
                                         name = 'd8_conv1')
    def call(self, X):
        x = self.b1_conv1(X[0])
        x = self.b1_conv2(x)
        b1_out = self.b1_pool1(x)
        
        x = self.b2_conv1(b1_out)
        x = self.b2_conv2(x)
        b2_out = self.b2_pool1(x)
        
        x = self.b3_conv1(b2_out)
        x = self.b3_conv2(x)
        x = self.b3_conv3(x)
        b3_out = self.b3_pool1(x)
        
        x = self.b4_conv1(b3_out)
        x = self.b4_conv2(x)
        x = self.b4_conv3(x)
        b4_out = self.b4_pool1(x)
        
        x = self.b5_conv1(b4_out)
        x = self.b5_conv2(x)
        x = self.b5_conv3(x)
        b5_out = self.b5_pool1(x)
        
        x = self.b6_conv1(b5_out)
        b6_out = self.b6_dropout1(x)
        x = self.b7_conv1(b6_out)
        b7_out = self.b7_dropout1(x)
        
        score32 = self.s32_conv1(b7_out)
        
        score16 = self.s16_conv1(b4_out)
        
        score8 = self.s8_conv1(b3_out)
        
        self.d32conv = self.d32_conv1(score32)
        
        self.d16conv = self.d16_conv1(self.d32conv + score16)
        
        self.d8conv = self.d8_conv1(self.d16conv + score8)
        
        self.pred = tf.argmax(self.d8conv, axis=-1, name="prediction")
        
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = X[1],
                                                                logits = self.d8conv)
        return self.pred, self.d8conv, self.loss


