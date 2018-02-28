#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:19:56 2018

@author: wahdan
"""

import numpy as np
import cntk
from cntk import Trainer
from cntk.learners import sgd
from cntk.ops import *
from cntk.io import *
from cntk.layers import *


# cntk.minus(1,cntk.abs(cntk.minus([1,2,3],[3,2,1]))).eval()


class SiamNet:
    
    def __init__(self):
        pass
            
    def build_network(self, input_shape):
        
        # w_init = RandomNormal(mean=0.0, stddev=1e-2, seed=None)
        # b_init = RandomNormal(mean=0.5, stddev=1e-2, seed=None)

        convnet = Sequential([
            Convolution((10, 10), 64, init=glorot_uniform(), pad=True, activation=relu),
            MaxPooling((2, 2), strides=(2, 2)),
            Convolution((7, 7), 128, init=glorot_uniform(), pad=True, activation=relu),
            MaxPooling((2, 2), strides=(2, 2)),
            Convolution((4, 4), 128, init=glorot_uniform(), pad=True, activation=relu),
            MaxPooling((2, 2), strides=(2, 2)),
            Convolution((4, 4), 256, init=glorot_uniform(), pad=True, activation=relu),
            Dense(4096, init=glorot_uniform(), activation=sigmoid)
        ])

        self.left_input = input_variable(input_shape)
        self.right_input = input_variable(input_shape)
        
        # encode each of the two inputs into a vector with the convnet
        encoded_l = convnet(self.left_input)
        encoded_r = convnet(self.right_input)

        merge = 1-abs(encoded_l-encoded_r)

        self.out = Dense(1,activation=sigmoid,init=glorot_uniform())(merge)

        return self.out

    def train(self, X1_train, X2_train, Y_train, X1_val, X2_val, Y_val,
              batch_size=128, epochs=10):
        assert X1_train.shape == X2_train.shape
        assert len(X1_train) == len(Y_train)
        assert X1_val.shape == X2_val.shape
        assert len(X1_val) == len(Y_val)

        if cntk.try_set_default_device(cntk.gpu(0)):
            print("GPU Training enabled")
        else:
            print("CPU Training :(")

        input_shape = (X1_train.shape[1], X1_train.shape[2], X1_train.shape[3])
        self.siamese_net = self.build_network(input_shape)

        lr_per_minibatch = cntk.learning_rate_schedule(0.1, cntk.UnitType.minibatch)
        pp = cntk.logging.ProgressPrinter()
            
        out = input_variable((1))
        loss = cntk.binary_cross_entropy(self.out,out)
        
        learner = cntk.adam(self.out.parameters, lr = lr_per_minibatch,momentum=0.9)
        trainer = cntk.Trainer(self.out, (loss,loss), [learner],[pp])
        
        cntk.logging.log_number_of_parameters(self.out)

        for epoch in range(epochs):
            # perm = np.random.permutation(len(Y_train))
            for i in range(0, len(Y_train), batch_size):
                max_n = min(i + batch_size, len(Y_train))
                # x1 = X1_train[perm[i:max_n]]
                # x2 = X2_train[perm[i:max_n]]
                # y = Y_train[perm[i:max_n]]
                x1 = X1_train[i:max_n]
                x2 = X2_train[i:max_n]
                y = Y_train[i:max_n]
                trainer.train_minibatch({self.left_input:x1,self.right_input:x2, out:y})
                pp.update_with_trainer(trainer,with_metric=True)
                print('.')
            pp.epoch_summary(with_metric=False)
            
    def predict(self, X1, X2):
            return 0
        
        
if __name__ == '__main__':
    print("Main")