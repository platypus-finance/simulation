# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:57:17 2020

@author: Lie Jen Houng
"""

import numpy as np

class Brownian:
    
    def __init__(self, s0, mu, sig, cv, step):
        # Check if the input is valid
        if not(len(s0) == len(mu) == len(sig) == len(cv) == len(cv[0])):
            raise Exception('Some input dimension is not {}'.format(len(s0)))
        self.num_asset = len(s0)
        self.s0 = s0
        self.mu = mu
        self.sig = sig
        self.cv = cv
        self.step = step
        self.h = [1/step]*self.num_asset

    def generate(self):
        z = np.matmul(np.linalg.cholesky(np.matrix(self.cv)),np.matrix(np.random.normal(0, 1, size=(self.num_asset, self.step))))
        s = np.zeros(shape=(self.step,self.num_asset))
        s[0,:] = self.s0
        for i in range(1,self.step):
            s[i,:] = s[i-1,:]*np.exp((self.mu-np.power(self.sig,2))*self.h+self.sig*np.sqrt(self.h)*np.array(z[:,i]).flatten())
        return np.transpose(s)
