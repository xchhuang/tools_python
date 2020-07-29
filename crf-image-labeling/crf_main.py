#!/usr/bin/python
# -*- coding:utf-8 -*-

import math
import os
import random
import numpy as np
import scipy.misc
import crf_model as crf_model


TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'


letterdict= {'e': 0, 't': 1, 'a': 2, 'i': 3, 'n': 4, 'o': 5, 's': 6, 'h': 7, 'r': 8, 'd': 9}    
alphabet = list('etainoshrd')
nb_classes = 10
iterations = 100
learning_rate = 0.1
batch_size = 32
mu = 0.9
v_Wf = 0
v_Wt = 0

with open(os.path.join(TRAIN_DIR,'train_words.txt'),'r') as traindoc:
    train_words=traindoc.read().split()

with open(os.path.join(TEST_DIR,'test_words.txt'),'r') as testdoc:
    test_words=testdoc.read().split()

# print (train_words)


X_train = []
X_test = []

for i in range(len(train_words)):
	X_train.append(np.loadtxt(os.path.join(TRAIN_DIR,"train_img"+str(i+1)+".txt")))

for i in range(len(test_words)):
    X_test.append(np.loadtxt(os.path.join(TEST_DIR,"test_img"+str(i+1)+".txt")))


Y_train = []
Y_test = []

for i in range(len(train_words)):
    Y_train.append(list(train_words[i]))
    
for i in range(len(test_words)):
    Y_test.append(list(test_words[i]))

# print (X_train[0].shape)
# print (len(Y_train[0]))

feature_dim = X_train[0].shape[1]

Wf = np.random.random((nb_classes, feature_dim))
# print (Wf.shape)

Wt = np.random.random((nb_classes, nb_classes))
# print (Wt.shape)

for i in range(iterations):

	# calculating the training and testing accuracy
	pred_test = crf_model.prediction_accuracy(Y_test, X_test, alphabet, Wf, Wt, len(Y_test))
	pred_train = crf_model.prediction_accuracy(Y_train, X_train, alphabet, Wf, Wt, len(Y_train))
	print (i, pred_train, pred_test)

	# train the crf using mini-batch sgd with nesterov momentum update and regularization
	# mini-batch
	index = np.random.randint(0, len(Y_train), batch_size)
	train, label = [], []
	for j in index:
		train.append(X_train[j])
		label.append(Y_train[j])

	# calculating the gradients
	Wf_grad = crf_model.feature_grad(label, train, alphabet, Wf, Wt)
	Wt_grad = crf_model.transition_grad(label, train, alphabet, Wf, Wt)

	# updating the weights using nesterov momentum sgd with regularization
	v_prev_Wf = v_Wf
	v_Wf = mu * v_Wf - learning_rate * Wf_grad
	Wf += -mu * v_prev_Wf + (1 + mu) * v_Wf - 2*0.01*Wf

	v_prev_Wt = v_Wt
	v_Wt = mu * v_Wt - learning_rate * Wt_grad
	Wt += -mu * v_prev_Wt + (1 + mu) * v_Wt - 2*0.01*Wt

pred = crf_model.predict(Y_test, X_test, alphabet, Wf, Wt)
print (pred)
