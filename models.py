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
        
        self.layer_names = [
                    'block3_pool', 
                    'block4_pool',   
                    'block5_pool',   
                ]
        self.layers = [self.vgg16.get_layer(name).output for name in self.layer_names]

        self.down_stack = Model(inputs=self.vgg16.input, outputs=self.layers)
        
        self.down_stack.trainable = True

        self.b6_conv1 = Conv2D(4096, 7, 1, padding = 'same',
                                    activation = 'relu',
                                    name='block6_conv1')     
        self.b6_drop = Dropout(0.5)
        
        self.b7_conv1 = Conv2D(4096, 1, 1, padding = 'same',
                                   activation = 'relu',
                                    name='block7_conv1')
        self.b7_drop = Dropout(0.5)
        
        self.s32_conv1 = Conv2D(self.n_classes, 1, 1, padding = 'same',
                                name = 's32_conv1') #200=n_classes

        self.d32_conv1 = Conv2DTranspose(self.n_classes, 64, 32, padding = 'same',
                                         name = 'd32_conv1')
    
    def build(self):
        self.inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = self.inputs
        s8, s16, s32 = self.down_stack(x)

        x = self.b6_conv1(s32)
        x = self.b6_drop(x)
        x = self.b7_conv1(x)
        x = self.b7_drop(x)
        
        score32 = self.s32_conv1(x)
        self.d32conv = self.d32_conv1(score32)
        
        self.pred = tf.argmax(self.d32conv, axis=-1, name="prediction")
        
        return tf.keras.Model(inputs=self.inputs, outputs=[self.pred, self.d32conv])




class FCN16s(FCN32s):
    def __init__(self, n_classes):
        super(FCN16s, self).__init__(n_classes)
        self.s32_conv1 = Conv2D(self.n_classes, 1, 1, padding = 'same',
                                name = 's32_conv1') #200=n_classes
        
        self.s16_conv1 = Conv2D(self.n_classes, 1, 1, padding = 'same',
                                name = 's16_conv1')

        self.d32_conv1 = Conv2DTranspose(self.n_classes, 4, 2, padding = 'same',
                                         name = 'd32_conv1')
        
        self.d16_conv1 = Conv2DTranspose(self.n_classes, 16, 16, padding = 'same',
                                         name = 'd16_conv1')


    def build(self):
        self.inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = self.inputs
        s8, s16, s32 = self.down_stack(x)

        x = self.b6_conv1(s32)
        x = self.b6_drop(x)
        x = self.b7_conv1(x)
        x = self.b7_drop(x)
        
        score32 = self.s32_conv1(x)
        
        score16 = self.s16_conv1(s16)

        self.d32conv = self.d32_conv1(score32)
        
        self.d16conv = self.d16_conv1(self.d32conv + score16)

        self.pred = tf.argmax(self.d16conv, axis=-1, name="prediction")
        
        return tf.keras.Model(inputs=self.inputs, outputs=[self.pred, self.d16conv])



class FCN8s(FCN32s):
    def __init__(self, n_classes):
        super(FCN8s, self).__init__(n_classes)
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

        self.d8_conv1 = Conv2DTranspose(self.n_classes, 16, 8, padding = 'same',
                                         name = 'd8_conv1')

    def build(self):
        self.inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = self.inputs
        s8, s16, s32 = self.down_stack(x)

        x = self.b6_conv1(s32)
        x = self.b6_drop(x)
        x = self.b7_conv1(x)
        x = self.b7_drop(x)
        
        score32 = self.s32_conv1(x)
        
        score16 = self.s16_conv1(s16)

        score8 = self.s8_conv1(s8)

        self.d32conv = self.d32_conv1(score32)
        
        self.d16conv = self.d16_conv1(self.d32conv + score16)

        self.d8conv = self.d8_conv1(self.d16conv + score8)

        self.pred = tf.argmax(self.d8conv, axis=-1, name="prediction")
        
        return tf.keras.Model(inputs=self.inputs, outputs=[self.pred, self.d8conv])
