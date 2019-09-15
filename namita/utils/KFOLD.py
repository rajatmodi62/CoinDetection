# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:47:21 2019

@author: hp
"""

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None) 

for train_index, test_index in kf.split('/content/drive/My Drive/Coin_dataset_31/'):
      print("Train:", train_index, "Validation:",test_index)
      X_train, X_test = X[train_index], X[test_index] 
      y_train, y_test = y[train_index], y[test_index]