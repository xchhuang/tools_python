#!/usr/bin/python
# -*- coding:utf-8 -*-

import math
import os
import random
import numpy as np
import math
import scipy.misc


def feature_potentials(word, feature_params):
# feature_params, C x n numpy array of feature parameter.
# word ,a w x n numpy array of feature vectors,
# word= X_test[i]
    return np.dot(word, np.transpose(feature_params))
    

def transition_potential (feat_pot1, feat_pot2, transition_params):
# Absorb node potential into a pairwise potential,
# for positions (t, t+1).
# output: (log) pairwise potential function.  (a table, e.g. array)
# Returns a C x C numpy array, where k is the size of the alphabet.
	# print (feat_pot1[:, np.newaxis].shape, transition_params.shape)
	tran_pot = transition_params + feat_pot1[:, np.newaxis]
	if feat_pot2 is not None:
		tran_pot += feat_pot2
	return tran_pot


def chain_potentials(word, feature_params, transition_params):
#Computes the clique potentials of the entire chain.
#Returns a (w-1) x C x C numpy array,
	phi = feature_potentials(word, feature_params)
	# print (phi[:-2])
	# Include the potentials of the last two nodes in the same clique
	transitions = [(node, None) for node in phi[:-2]] + [(phi[-2], phi[-1])]
	# print (transitions)
	#transitions = [(node, None) for node in phi[:]]
	psi = [transition_potential(node1, node2, transition_params) for node1, node2 in transitions]
	return np.array(psi)


def message_passing (psi):
# Message passing algorithm
# input: (log-) potential
# outputs: forward/backward messages
	# Backward messages
	back = []
	prev_msgs = np.zeros(psi.shape[1])
	# print (prev_msgs.shape)
	# print (psi)
	# print 
	# print (psi[:-1])
	for pairs in psi[:0:-1]:
		# print (pairs)
		# print 
		# print (prev_msgs)
		
		message = scipy.misc.logsumexp(pairs + prev_msgs, axis=1)
		# print ((pairs + prev_msgs))
		# print
		# print (message)
		# break
		back.append(message)
		prev_msgs += message

	# Forward messages
	fwd = []
	prev_msgs = np.zeros(psi.shape[1])
	for pairs in psi[:-1]:
		message = scipy.misc.logsumexp(pairs + prev_msgs[:, np.newaxis], axis=0)
		fwd.append(message)
		prev_msgs += message

	return (np.array(back), np.array(fwd))


def beliefs(word, feature_params, transition_params):
# Returns a numpy array of size (w-1) x k x k,
    psi = chain_potentials(word,feature_params,transition_params)
    delta_bwd, delta_fwd = message_passing(psi)

    k = delta_fwd.shape[1]
    delta_fwd = np.concatenate(([np.zeros(k)], delta_fwd))
    delta_bwd = np.concatenate((delta_bwd[::-1], [np.zeros(k)]))
    belief = psi + delta_fwd[:, :, np.newaxis] + delta_bwd[:, np.newaxis]

    return np.array(belief)


def pairwise_prob(belief): 
# pairwise marginal probabilities.
    return np.exp(belief - scipy.misc.logsumexp(belief, axis=(1,2))[:, np.newaxis, np.newaxis])

def single_prob(pairwise_p):
# singleton marginal probabilities.
	# print (pairwise_p)
	a = np.sum(pairwise_p, axis=2)
	# print (a)

	b = np.sum(pairwise_p[-1], axis=0) # Last character in the word
	# print (b)

	return np.concatenate((a, b[np.newaxis, :]))


def predict_word(single_p, alphabet):
# Returns a list of predicted characters of a word.
# Parameters:
#    - single_p, a w x k numpy array of singleton marginal probabilities,
#      where w is the word length and k is the size of the alphabet; and
#    - alphabet, a list of all possible character labels.
    indices = np.argmax(single_p, axis=1)
    
    return [alphabet[i] for i in indices]

def predict(Y_train,X_train,alphabet,feature_params,transition_params):
# Returns a list of predictions, where each prediction is a list of predicted character labels of a word.
    predictions = []
    for i in range(len(Y_train)):
        belief = beliefs(X_train[i],feature_params,transition_params)
        pairwise_p = pairwise_prob(belief)
        single_p = single_prob(pairwise_p)
        predictions.append(predict_word(single_p, alphabet))

    return predictions


def prediction_accuracy(Y_test, X_test, alphabet, feature_params, transition_params, n):
# Returns accuracy of the sequence predcition task
# Limiting to n words
    predictions= predict(Y_test,X_test,alphabet,feature_params,transition_params)
    total=0
    correct=0
    for i in range(n):
        total+=len(Y_test[i])
        count=0
        for j in range(len(Y_test[i])):
            if predictions[i][j]==Y_test[i][j]:
                count+=1
        correct+= count
    return 100*correct/total


def feature_grad(Y_train,X_train,alphabet,feature_params,transition_params):
# Returns a flattened k x n numpy array,where k is the size of the alphabet and n is the length of the feature vector.
    # Initialize a state gradient table of size k x n with zeros
    gradient = np.zeros((len(alphabet), len(X_train[0][1])))

    for i in range(len(Y_train)):
        belief = beliefs(X_train[i], feature_params, transition_params)
        pairwise_p = pairwise_prob(belief)
        single_p = single_prob(pairwise_p)
        for v, c, p in zip(X_train[i], Y_train[i], single_p):
            for i in range(gradient.shape[0]): # possible labels
                for j in range(gradient.shape[1]): # features
                    indicator = 0
                    if c == alphabet[i]:
                        indicator = 1
                    gradient[i][j] += (indicator - p[i]) * v[j]
    
    gradient /= len(Y_train)
    return -gradient
    # return np.ndarray.flatten((gradient))


def transition_grad(Y_train,X_train,alphabet,feature_params,transition_params):
# Returns a flattened k x k numpy array, where k is the size of the alphabet.

    # Initialize a transition gradient table of size k x k with zeros
    gradient = np.zeros((len(alphabet), len(alphabet)))

    for i in range(len(Y_train)):
        belief = beliefs(X_train[i], feature_params, transition_params)
        pairwise_p = pairwise_prob(belief)
        label_pairs = list(zip([None] + Y_train[i], Y_train[i] + [None]))[1:-1]

        for (label1, label2), p in zip(label_pairs, pairwise_p):
            for i in range(gradient.shape[0]):
                for j in range(gradient.shape[1]):
                    indicator = 0
                    if label1 == alphabet[i] and label2 == alphabet[j]:
                        indicator = 1
                    gradient[i][j] += indicator - p[i][j]

    gradient /= len(Y_train)
    return -gradient
    # return np.ndarray.flatten((gradient))
