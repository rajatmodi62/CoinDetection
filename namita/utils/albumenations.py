# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:26:08 2019

@author: hp
"""

import Augmentor
p = Augmentor.Pipeline("C:/Users/hp/Desktop/Coin_dataset_31/train")

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.sample(100)

p.process()