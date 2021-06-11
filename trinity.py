# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:52:08 2020

@author: Erik Lie
"""

from math import log
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc

# Constants
ETH_constant = 365 * 24 * 60 * 60 / 13

a = 0.4
b = 1

alpha = 5
gamma = 1

delta_max = 0.000000286
    
h = 0.003
rr = 0.2
print_ind = False

# Custom Functions

def deannualized(x):
    return (1+x)**(1 / ETH_constant) - 1

def f(x):
    if x >= b:
        return 0
    elif x >= a:
        return (x-b)**2/(2*(b-a))
    else:
        return -x + (a+b)/2

def g(x):
    return gamma / x**alpha

def add_or_assign(name, dictionary, amount):
    if name in dictionary.keys():
        dictionary[name] += amount
    else:
        dictionary[name] = amount
        
def arb_profit(from_token, to_token):
    eqm = (from_token.asset() * from_token.price + to_token.asset() * to_token.price) / (from_token.liab() * from_token.price + to_token.liab() * to_token.price)
    profit = -(from_token.liab() * from_token.price * (f(eqm) - f(from_token.cov_ratio())) + to_token.liab() * to_token.price * (f(eqm) - f(to_token.cov_ratio())))
    return (eqm, profit)
        
class users:
    
    def __init__(self, name):
        self.name = name
        self.wallet = {}
        self.lp_token = {}
        
    @classmethod
    def get_instances(cls):
        result = []
        for obj in gc.get_objects():
            if isinstance(obj, cls):
                result.append(obj)
        return result
        
    def deposit(self, pool, token, amount):
        pool.deposit(self, token, amount)
    
    def withdraw(self, pool, token, amount):
        pool.withdraw(self, token, amount)
        
    def swap(self, pool, from_token, to_token, from_amount):
        pool.swap(self, from_token, to_token, from_amount)
        
class tokens:
    
    def __init__(self, name):
        self.name = name
        self.cash = 0
        self.deposit = 0
        self.lp_tokens = 0
        self.price = 0
        self.true_price = 0
        self.temp_withdraw = 0
        self.temp_swap = 0
        self.hf_sys = 0
        self.sf = 0
    
    @classmethod
    def get_instances(cls):
        result = []
        for obj in gc.get_objects():
            if isinstance(obj, cls):
                result.append(obj)
        return result
        
    def asset(self):
        return self.cash
    
    def liab(self):
        return self.deposit
    
    def cov_ratio(self):
        return 0 if self.liab() == 0 else self.asset()/self.liab()
    
    def quk_ratio(self):
        return 0 if self.liab() == 0 else self.asset()/self.liab()
    
    def temp_asset(self):
        return self.cash + self.temp_swap - self.temp_withdraw
    
    def temp_liab(self):
        return self.deposit - self.temp_withdraw
    
    def temp_cov_ratio(self):
        return 0 if self.temp_liab() == 0 else self.temp_asset()/self.temp_liab()
    
    def temp_quk_ratio(self):
        return 0 if self.temp_liab() == 0 else self.temp_asset()/self.temp_liab()
        
    def withdraw_fees(self):
        r_old = self.cov_ratio()
        r_new = self.temp_cov_ratio()
        return 0 if r_old == r_new else ((1-r_old)*f(r_new)-(1-r_new)*f(r_old))/(r_new - r_old)
    
    def price_slippage(self):
        r_old = self.cov_ratio()
        r_new = self.temp_cov_ratio()
        return 0 if r_old == r_new else (f(r_new) - f(r_old)) / (r_new - r_old)
    
class pool:
    
    def __init__(self):
        return
    
    def deposit(self, user, token, amount):
        
        # cash transfer
        add_or_assign(token.name, user.wallet, -amount)
        token.cash += amount

        # ....................................................................
        # ..             User              ..               Pool            ..
        # ....................................................................
        # ..   - cash                      ..   + cash                      ..
        # ..   + deposit                   ..   + liability                 ..
        # ....................................................................
        
        # token minting
        x = amount * (1 if token.deposit == 0 else token.lp_tokens/token.deposit)
        add_or_assign(token.name, user.lp_token, x)
        token.lp_tokens += x
        token.deposit += amount
        
        if print_ind == True:
            print('Sucessfully deposited {0} {1} from Pool for {2}'.format(amount, token.name, user.name))
        
    def withdraw(self, user, token, perc):
        
        # Calc the actual withdrawal amount
        amount = user.lp_token[token.name] * perc * token.deposit / token.lp_tokens
        
        # Set temporary variable
        token.temp_withdraw = amount
        
        # withdrawal fees
        w_fees = -amount * token.withdraw_fees()
        
        # cash transfer
        add_or_assign(token.name, user.wallet, amount - w_fees)
        token.cash -= amount - w_fees
        
        # ....................................................................
        # ..             User              ..               Pool            ..
        # ....................................................................
        # ..   + cash - withdrawal_fees    ..   - cash + withdrawal_fees    ..
        # ..   - deposit                   ..   - liability                 ..
        # ....................................................................
        
        # token burning
        x = user.lp_token[token.name] * perc
        add_or_assign(token.name, user.lp_token, -x)
        token.lp_tokens -= x
        token.deposit -= amount
        
        # Reset temporary variable
        token.temp_withdraw = 0
        
        if print_ind == True:
            print('Sucessfully withdrew {0} {1} from Pool for {2}. Fees charge: {3}.'.format(amount, token.name, user.name, w_fees))
        
    def swap(self, user, from_token, to_token, from_amount):

        # Calculate the ideal_to_amount
        to_amount = from_amount * (from_token.price / to_token.price)
        
        # Set temporary variables
        from_token.temp_swap = from_amount
        to_token.temp_swap = -to_amount
        
        # Calculate swapping slippage
        ss = from_token.price_slippage() - to_token.price_slippage()
        sf = to_amount * ss
                
        # Haircut
        hf = (to_amount - sf) * h
        hf_lp = hf * (1-rr)

        # Transfer
        act_to_amount = to_amount - sf - hf
        
        add_or_assign(from_token.name, user.wallet, -from_amount)
        from_token.cash += from_amount
        
        add_or_assign(to_token.name, user.wallet, act_to_amount)
        to_token.cash -= act_to_amount
        
        # Some haircut goes to LPs
        to_token.deposit += hf_lp        
                        
        # ...............................................................................................
        # ..             User              ..               Pool            ..           LPs           ..
        # ...............................................................................................
        # ..   - from_amount               ..   + from_amount               ..                         ..
        # ..   + to_amount - sf - hf       ..   - to_amount + sf + hf       ..        + hf_lp          ..
        # ...............................................................................................
                
        # Reset temporary variable
        from_token.temp_swap = 0
        to_token.temp_swap = 0
        
        # Record the haircut and slippage
        to_token.hf_sys += hf - hf_lp
        to_token.sf += sf    
        
        if print_ind == True:
            print('Sucessfully swapped {0} {1} from {2} {3} for {4}. Fees charged: {5}'.format(act_to_amount,to_token.name,from_amount,from_token.name, user.name, sf + hf))