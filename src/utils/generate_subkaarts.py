#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:48:25 2023

@author: mli1
"""

def generate_subkaarts(big_kaarts):
    train_subs = []
    val_subs = []
    test_subs = []
    for kaart in big_kaarts:
        kaart = str(kaart)
        train_subs = train_subs + [kaart + '_5-6', kaart + '_1-2'] 
        val_subs = val_subs + [kaart + '_3-4']
        test_subs = test_subs + [kaart + '_7-8']
    all_subs = [train_subs, val_subs, test_subs]
    return all_subs
