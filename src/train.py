#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:27:40 2018

@author: wahdan
"""

from SiamNet import SiamNet
from generate_data import generate_train_test_data

X1_train, X2_train, Y_train, X1_val, X2_val, Y_val = generate_train_test_data('d:/att_faces')

model = SiamNet()

model.train(X1_train, X2_train, Y_train, X1_val, X2_val, Y_val, batch_size=128, epochs=1)

#model.save()

#model.kill()